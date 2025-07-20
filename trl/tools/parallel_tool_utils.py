"""
Batch-oriented tool-calling driver for vLLM.

Key ideas
----------
* Keep one list element per prompt (`contexts[i]`, `tokens_left[i]`, …).
* At each step:
    1.  Send **only the unfinished** contexts to vLLM in **one** batched
        `/generate` request.
    2.  Append the model output to the corresponding context.
    3.  Detect tool calls in every new chunk.
    4.  Execute the tool(s) in parallel processes, insert <tool_response>…, and mark
        that prompt for another generation pass.
    5.  If a prompt finishes (no tool call or `tokens_left==0`) produce its
        final answer and mark it *done*.
* Loop until every prompt is *done*.

"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
import trl
import os
import traceback
import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy

from .python_and_retriever import (
    google_retriever,
    python_interpreter,
    remove_python_prefix,
)
from ..trainer.grpo_config import GRPOConfig
from .python_and_retriever import google_retriever,python_interpreter, remove_python_prefix
from .async_tool_utils import _detect_tool_use
from ..import_utils import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


################################################################################
# vLLM helpers
################################################################################


def _make_tool_inference_requests_batch(
    prompts: List[str],
    max_tokens: int,
    args: GRPOConfig,
    stop: Optional[List[str]] | None = None,
    llm: LLM = None,
) -> Tuple[List[List[int]], List[str]]:
    """
    Submit **N** prompts in one request and return (ids_list, text_list) with
    matching order.
    """
    if args.vllm_mode == "server":
        url = f"http://{args.vllm_server_host}:{args.vllm_server_port}/generate/"
        resp = requests.post(
            url,
            json={
                "prompts": prompts,
                "n": 1,
                "repetition_penalty": args.repetition_penalty,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": -1 if args.top_k is None else args.top_k,
                "min_p": 0.0 if args.min_p is None else args.min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": args.vllm_guided_decoding_regex,
                "stop": stop,
            },
            timeout=600,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM request failed: {resp.status_code}: {resp.text}")

        j = resp.json()
        return j["completion_ids"], j["completion_text"]  # type: ignore[return-value]

    elif args.vllm_mode == "colocate":
        assert llm is not None, "Require llm for tool inference"
        sampling_params = SamplingParams(
            n=1,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1 if args.top_k is None else args.top_k,
            min_p=0.0 if args.min_p is None else args.min_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=max_tokens,
            stop=stop,
        )

        outputs = llm.generate(prompts, sampling_params)
        return (
            [out.outputs[0].token_ids for out in outputs],
            [out.outputs[0].text for out in outputs],
        )
        
    else:
        raise ValueError(f"Unsupported vllm_mode: {args.vllm_mode}")

################################################################################
# Parallel tool executor
################################################################################ 
    
def _execute_tool_job(args):
    """Worker for executing tool calls (Python or retrieval)."""
    exec_error = False
    global_i, detect, context, args_, tokenizer = args
    try:
        tool_type = detect["tool_type"]
        if tool_type == "python":
            # print("Calling Python Interpreter")
            if isinstance(detect["tool_argument"], str):
                code = detect["tool_argument"]
            else:
                code = detect["tool_argument"]["code"]
                
            try:
                new_code, exec_out = python_interpreter(code)
            except Exception as e:
                print(f"[ERROR] Python execution failed for idx={global_i}:\n{code}\n{e}")
                exec_out = ""
                new_code = code
                exec_error = True

            new_python_code = json.dumps({
                "name": "python",
                "arguments": {"code": new_code}
            }, ensure_ascii=False)
            
            python_start_idx = detect["tool_start_idx"]
            context = (
                context[:python_start_idx]
                + f"{args_.tool_tokens[0]}\n" + new_python_code + f"\n{args_.tool_tokens[1]}\n"
                f"{args_.result_tokens[0]}\n" + exec_out + f"\n{args_.result_tokens[1]}\n"
            )
            print(f"Python Interpreter Finished. Result: {exec_out[:30]}")
            return global_i, tool_type, context, exec_error

        elif tool_type in ("retrieval", "retrieve"):
            # print("Calling Web Retriever")
            try:
                result = google_retriever(**detect["tool_argument"])
                result_text = result.response if result else ""
                if not result:
                    exec_error = True
            except Exception as e:
                print(f"[ERROR] Retrieval failed for idx={global_i}: {e}")
                result_text = ""
                exec_error = True

            context = (
                context + # Ctxt already finishes with \n
                f"{args_.result_tokens[0]}\n" + result_text + f"\n{args_.result_tokens[1]}\n"
            )
            print(f"Web Retriever Finished. Result: {result_text[:30]}")
            return global_i, tool_type, context, exec_error
        else:
            # Sometimes the model generates an invalid tool type.
            raise ValueError(f"Unknown tool type: {tool_type}")

    except Exception as e:
        print(f"[ERROR] Tool job failed for idx={global_i}: {e}")
        return global_i, None, context, True


################################################################################
# Core driver
################################################################################

def generate_with_tool_batch(
    prompts: List[str],
    args: GRPOConfig,
    llm: LLM = None,
    tokenizer = None,
    budget_force_saving: int = 0,
    max_saving: int = 1,
    save_last_k: int = 0,
    saving_prompt = None,
    move_context_to_prompt: bool = False,
    use_max: bool = False,
    return_saving: bool = True,
) -> Tuple[List[str], List[str], List[Dict[str, int]]]:
    """
    Parameters
    ----------
    prompts
        Raw user questions / conversations (each already contains any system /
        role prefixes you want).
        
    Returns
    -------
    final_contexts
        The fully-expanded conversation for every prompt (tool results +
        answers).
    stats
        Per-prompt statistics: number of python / retrieval / saving calls, &
        total generated tokens.
    """
    N = len(prompts)
    if budget_force_saving:
        if saving_prompt == "summary_maybe":
            force_saving_str = '\n' + args.tool_tokens[0] + '\n{"name": "save", "arguments": {"summary": "'
        else:
            force_saving_str = '\n' + args.tool_tokens[0] + '\n{"name": "pass", "arguments": {"context": "'    
        force_saving_len = len(tokenizer(force_saving_str).input_ids)
        budget_force_saving_net = budget_force_saving - force_saving_len
        args.max_completion_length -= budget_force_saving
        is_forced = [False] * N # To track if last saving was forced

    contexts = prompts[:]
    prompts_len = [len(p) for p in prompts]
    saving_prompt_per_prompt = [[] for _ in range(N)]
    saving_completions_per_prompt = [[] for _ in range(N)]
    tokens_left = [args.max_completion_length] * N
    done = [False] * N
    stats = [
        {"generated_tokens": 0, 
        "num_python": 0, 
        "num_retrieval": 0, 
        "num_saving": 0,
        "num_saving_forced": 0,        
        "num_tool_detect_error": 0, 
        "num_python_exec_error": 0,
        "num_retriever_exec_error": 0,
        "num_saving_exec_error": 0,
        "num_saving_invalid_use": 0,
        } for _ in range(N)
    ]
    malformed_tool_call_str = f"\n{args.result_tokens[0]}\n{args.result_tokens[1]}\n"
    malformed_tool_call_len = len(tokenizer(malformed_tool_call_str).input_ids)

    step = 0
    while not all(done):
        step += 1
        active_idx = [i for i, d in enumerate(done) if not d and tokens_left[i] > 0]
        if len(active_idx) == 0: break
        active_prompts = [contexts[i] for i in active_idx]
        tokens_left_active = [tokens_left[i] for i in active_idx]

        if use_max:
            # import pdb; pdb.set_trace()
            # ~2x faster with 1x saving (but higher OOM risk)
            comp_ids_list, comp_text_list = _make_tool_inference_requests_batch(
                active_prompts, # Technically could always append <think> for Qwen3 to save 1 token
                max_tokens=max(tokens_left_active),
                args=args,
                stop=[args.tool_tokens[1], args.eos_token],
                llm=llm,
            )
            comp_ids_list = [
                comp_ids[:max_tokens] for comp_ids, max_tokens in zip(comp_ids_list, tokens_left_active)
            ]
            comp_text_list = tokenizer.batch_decode(comp_ids_list)
        else:
            comp_ids_list, comp_text_list = _make_tool_inference_requests_batch(
                active_prompts,
                max_tokens=min(tokens_left_active),
                args=args,
                stop=[args.tool_tokens[1], args.eos_token],
                llm=llm,
            )

        # Update contexts and detect tool use
        tool_jobs = []
        for local_i, global_i in enumerate(active_idx):
            comp_ids = comp_ids_list[local_i]
            comp_ids_len = len(comp_ids)
            comp_text = comp_text_list[local_i]
            stop_token = tokenizer.decode(comp_ids[-1])
            if not(comp_text.endswith(stop_token)):
                comp_text += stop_token
            contexts[global_i] += comp_text
            stats[global_i]["generated_tokens"] += comp_ids_len
            tokens_left[global_i] -= comp_ids_len

            if stop_token == args.eos_token:
                done[global_i] = True
                continue

            detect = _detect_tool_use(contexts[global_i])

            if detect["error_tool_use"]:
                stats[global_i]["num_tool_detect_error"] += 1
                print(f"[{global_i}] malformed tool call – skipping tool use.")
                if tokens_left[global_i] > malformed_tool_call_len:
                    contexts[global_i] += malformed_tool_call_str
                    tokens_left[global_i] -= malformed_tool_call_len
                    stats[global_i]["generated_tokens"] += malformed_tool_call_len
                else:
                    done[global_i] = True
                continue

            if not detect["tool_use"]:
                if tokens_left[global_i] > 0: continue
                if budget_force_saving and (stats[global_i]["num_saving"] < max_saving):
                    if not(is_forced[global_i]): # Only force once per ctxt
                        # Force saving tool call
                        contexts[global_i] += force_saving_str
                        tokens_left[global_i] += budget_force_saving_net
                        stats[global_i]["generated_tokens"] += force_saving_len
                        is_forced[global_i] = True
                        stats[global_i]["num_saving_forced"] += 1
                        continue
                    else:
                        # In case it ran out of context while calling the tool
                        detect = _detect_tool_use(contexts[global_i], prompt_idx=prompts_len[global_i], force_end_tag=False)
                        if not(detect["tool_use"] and detect["tool_type"] in ["saving", "save", "pass"]):
                            continue # Will be marked as done as no tokens left
                        else:
                            print("Continuing with broken saving tool call.")
                elif save_last_k:
                    # Here it is fine to end directly as it has used the full sequence; no extra budget
                    if stats[global_i]["num_saving"] >= max_saving:
                        continue # Will be marked as done as no tokens left

                    # If it has saved before, then it starts with saving part which we may want to move to the prompt
                    toks = tokenizer(contexts[global_i][prompts_len[global_i]:]).input_ids                    
                    if (move_context_to_prompt) and (stats[global_i]["num_saving"] > 0):
                        saving_txt_prev = tokenizer.decode(toks[:save_last_k])
                        saving_prompt_per_prompt[global_i].append(contexts[global_i][:prompts_len[global_i]] + saving_txt_prev)
                        saving_completions_per_prompt[global_i].append(tokenizer.decode(toks[save_last_k:]))
                    else:
                        saving_prompt_per_prompt[global_i].append(contexts[global_i][:prompts_len[global_i]])
                        saving_completions_per_prompt[global_i].append(contexts[global_i][prompts_len[global_i]:])

                    saving_txt_next = tokenizer.decode(toks[-(save_last_k-2):]) # Subtract 2 for think & \n
                    add = args.thinking_tokens[0] + "\n" + saving_txt_next
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add

                    tokens_left[global_i] = args.max_completion_length
                    stats[global_i]["num_saving"] += 1
                    continue
                else:
                    done[global_i] = True
                    continue

            tool_type = detect["tool_type"]
            if tool_type in ["saving", "save", "pass"]:
                if stats[global_i]["num_saving"] >= max_saving:
                    # Could mark it as done but ideally it never lands here by changing the system prompt
                    # to clarify when it cannot use it
                    stats[global_i]["num_saving_invalid_use"] += 1
                    continue
                stats[global_i]["num_saving"] += 1
                if budget_force_saving: # Reset is_forced as used now
                    is_forced[global_i] = False

                if "summary" in detect.get("tool_argument", {}):
                    summary = detect["tool_argument"]["summary"]
                elif "context" in detect.get("tool_argument", {}):
                    summary = detect["tool_argument"]["context"]
                else:
                    summary = ""
                    stats[global_i]["num_saving_exec_error"] += 1

                saving_completions_per_prompt[global_i].append(contexts[global_i][prompts_len[global_i]:] + args.eos_token)
                saving_prompt_per_prompt[global_i].append(contexts[global_i][:prompts_len[global_i]])

                if saving_prompt is None:
                    add = f"{args.thinking_tokens[0]}\n{summary}\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add
                elif saving_prompt == "debug":
                    toks = 0
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]]
                elif saving_prompt == "context":
                    add = f"\n\nContext:\n{summary}<|im_end|>\n<|im_start|>assistant\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i] - len("<|im_end|>\n<|im_start|>assistant\n")] + add
                    # Commented out below to allow full tokens as removed system prompt & input tokens don't matter as much as output ones
                    # toks = len(tokenizer(add).input_ids) - 5 # <|im_end|>\n<|im_start|>assistant\n
                elif saving_prompt == "context_maybe":
                    # Context that you can use, but may be incorrect:
                    add = f"\n\nContext from your prior reasoning attempt that may be useful but could also be wrong:\n{summary}<|im_end|>\n<|im_start|>assistant\n"
                    if stats[global_i]["num_saving"] == max_saving:
                        user_start = contexts[global_i].rfind("<|im_start|>user\n") # Remove system prompt
                    else:
                        user_start = 0
                    contexts[global_i] = contexts[global_i][user_start:prompts_len[global_i]-len("<|im_end|>\n<|im_start|>assistant\n")] + add
                elif saving_prompt == "context_system":
                    # Context that you can use, but may be incorrect:
                    system = 'When context is passed to you, evaluate its correctness first and then use it if correct, below is an example.\n<|im_start|>user\nThere exist real numbers $x$ and $y$, both greater than 1, such that $\log_x\left(y^x\\right)=\log_y\left(x^{4y}\\right)=10$. Find $xy$.\n\nContext:\nMaybe convert logarithmic equations into exponential form to make them easier to handle. Via logarithm power rule log_x(y^x) = x * log_x(y) = 10. Noting that log_x(y) is the same as 1 / log_y(x). Maybe try express log_x(y) in terms of something else.<|im_end|>\n<|im_start|>assistant\n<think>\nOkay, so I need to first evaluate whether the context is correct and then use it to solve the problem.\n\nThe context suggests converting logarithmic equations into exponential form and using the logarithm power rule, which is a valid approach. It also notes that log_x(y) can be expressed in terms of log_y(x), which is correct. It suggests expressing log_x(y) in terms of something else.\n\nWait, do we really need to express log_x(y) in terms of something else? Maybe we can instead first use the logarithm power rule for the second equation as well. Let me do that...'
                    add = f"\n\nContext:\n{summary}<|im_end|>\n<|im_start|>assistant\n"
                    if stats[global_i]["num_saving"] == max_saving:
                        prepend = "<|im_start|>system\n" + system + "<|im_end|>\n"
                        user_start = contexts[global_i].rfind("<|im_start|>user\n") # Remove system prompt
                    else:
                        prepend = ""
                        user_start = 0
                    contexts[global_i] = prepend + contexts[global_i][user_start:prompts_len[global_i]-len("<|im_end|>\n<|im_start|>assistant\n")] + add                    
                elif saving_prompt == "summary_maybe":
                    add = f"\n\nSummary of your prior reasoning attempt that may be useful but could also be wrong:\n{summary}<|im_end|>\n<|im_start|>assistant\n"
                    if stats[global_i]["num_saving"] == max_saving:
                        user_start = contexts[global_i].rfind("<|im_start|>user\n") # Remove system prompt
                    else:
                        user_start = 0
                    contexts[global_i] = contexts[global_i][user_start:prompts_len[global_i]-len("<|im_end|>\n<|im_start|>assistant\n")] + add
                elif saving_prompt == "special_before_thinking":
                    add = f"{args.saving_tokens[0]}\n{summary}\n{args.saving_tokens[1]}\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add
                elif saving_prompt == "special_after_thinking":
                    add = f"{args.thinking_tokens[0]}\n{args.saving_tokens[0]}\n{summary}\n{args.saving_tokens[1]}\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add
                else:
                    raise ValueError(f"Unknown saving_prompt: {saving_prompt}")

                tokens_left[global_i] = args.max_completion_length # - toks # Not subtracting toks, see above
                if budget_force_saving and (stats[global_i]["num_saving"] == max_saving):
                    tokens_left[global_i] += budget_force_saving # If already used last saving, give it rest of seq len
            else:
                if tool_type not in ("python", "retrieval", "retrieve"):
                    print(f"[{global_i}] Unknown tool type: {tool_type}. Skipping tool use.")
                    stats[global_i]["num_tool_detect_error"] += 1
                    if tokens_left[global_i] > malformed_tool_call_len:
                        contexts[global_i] += malformed_tool_call_str
                        tokens_left[global_i] -= malformed_tool_call_len
                        stats[global_i]["generated_tokens"] += malformed_tool_call_len
                    else:
                        done[global_i] = True
                    continue
                tool_jobs.append((global_i, deepcopy(detect), contexts[global_i], args, tokenizer))

        if tool_jobs:
            tool_start = time.time()
            n_tool_jobs = len(tool_jobs)
            with ProcessPoolExecutor(max_workers=min(n_tool_jobs, os.cpu_count())) as executor:
                results = list(executor.map(_execute_tool_job, tool_jobs))

            for global_i, tool_type, updated_context, exec_error in results:
                toks = tokenizer(updated_context).input_ids
                if (n_toks := len(toks)) >= args.max_completion_length:
                    done[global_i] = True
                    updated_context = tokenizer.decode(toks[:args.max_completion_length])
                    n_toks = args.max_completion_length
                tokens_left[global_i] = args.max_completion_length - n_toks
                contexts[global_i] = updated_context
                if tool_type == "python":
                    stats[global_i]["num_python"] += 1
                    if exec_error:
                        stats[global_i]["num_python_exec_error"] += 1
                elif tool_type in ("retrieval", "retrieve"):
                    stats[global_i]["num_retrieval"] += 1
                    if exec_error:
                        stats[global_i]["num_retriever_exec_error"] += 1

            tool_duration = time.time() - tool_start
            print(f"[Time] {tool_duration:.2f} seconds for {n_tool_jobs} parallel tool calls. Avg {tool_duration / n_tool_jobs:.2f} seconds.")

    prompts_to_return: List[str] = []
    completions_to_return: List[str] = []

    if return_saving:
        for i in range(N):
            prompts_to_return.extend(saving_prompt_per_prompt[i])
            completions_to_return.extend(saving_completions_per_prompt[i])

    if save_last_k and move_context_to_prompt:
        for i, context in enumerate(contexts):
            # these have saving hence need to adjust the prompt added
            if stats[i]["num_saving"] > 0:
                toks = tokenizer(context[prompts_len[i]:]).input_ids
                saving_txt = tokenizer.decode(toks[:save_last_k])
                prompts_to_return.append(context[:prompts_len[i]] + saving_txt)
                completions_to_return.append(tokenizer.decode(toks[save_last_k:]))
            else:
                prompts_to_return.append(context[:prompts_len[i]])
                completions_to_return.append(context[prompts_len[i]:])
    else:
        for i, context in enumerate(contexts):
            prompts_to_return.append(context[:prompts_len[i]])
            completions_to_return.append(context[prompts_len[i]:])

        if move_context_to_prompt:
            for i in range(len(prompts_to_return)):
                p = prompts_to_return[i] + completions_to_return[i]
                split_idx = p.rfind("<|im_end|>\n<|im_start|>assistant\n") + len("<|im_end|>\n<|im_start|>assistant\n")
                prompts_to_return[i] = p[:split_idx]
                completions_to_return[i] = p[split_idx:]

    args.max_completion_length += budget_force_saving
    # if any(len(tokenizer(c).input_ids) > args.max_completion_length for c in completions_to_return):
    #     import pdb; pdb.set_trace()
    # Tokenizers are not deterministic:
    # When the model generates in vLLM it may generate these tokens:
    # ['Ġto', 'ĠFG', 'NM', 'Ġto']
    # when they are decoded and provided as completions text it turns into
    #  to FGNM to
    # when they are retokenized it turns into
    # ['Ġto', 'ĠF', 'GN', 'M', 'Ġto']
    # hence the below retruncates to avoid > max seq len.
    # The code above can be used for detection.
    for i, c in enumerate(completions_to_return):
        toks = tokenizer(c).input_ids
        if (toks_len := len(toks)) > args.max_completion_length:
            completions_to_return[i] = tokenizer.decode(toks[:args.max_completion_length])
            toks_len_new = len(tokenizer(completions_to_return[i]).input_ids)
            print(f"[WARNING] Truncated completion {i} to fit max length: {toks_len} -> {toks_len_new}")
        # Note that it could still be smaller and not end with an EOS which is handled later via padding

    # print("Saving stats:", [stats[i]["num_saving"] for i in range(N)])
    print("Saving stats avg:", sum(stats[i]["num_saving"] for i in range(N)) / N)
    print("Max Saving %:", sum(stats[i]["num_saving"] == max_saving for i in range(N)) / N * 100)
    # print("Generated tokens:", [stats[i]["generated_tokens"] for i in range(N)])
    print("Generated tokens avg:", sum(stats[i]["generated_tokens"] for i in range(N)) / N)
    print("Steps total:", step)
    return prompts_to_return, completions_to_return, stats
