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
    force_saving: bool = False,
    max_saving: int = 15,
    save_last_k: Optional[int] = 256,
    saving_prompt = None,
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
    contexts = prompts[:]
    prompts_len = [len(p) for p in prompts]
    saving_prompt_per_prompt = [[] for _ in range(N)]
    saving_completions_per_prompt = [[] for _ in range(N)]
    num_steps_per_prompt = [0] * N
    tokens_left = [args.max_completion_length] * N
    done = [False] * N
    stats = [
        {"generated_tokens": 0, 
        "num_python": 0, 
        "num_retrieval": 0, 
        "num_saving": 0, 
        "num_tool_detect_error": 0, 
        "num_python_exec_error": 0,
        "num_retriever_exec_error": 0,
        "num_saving_exec_error": 0,
        } for _ in range(N)
    ]
    malformed_tool_call_str = f"\n{args.result_tokens[0]}\n{args.result_tokens[1]}\n"
    malformed_tool_call_len = len(tokenizer(malformed_tool_call_str).input_ids)
    saving_force_str = '\n' + args.tool_tokens[0] + '\n{"name": "pass", "arguments": {"context": "'
    saving_force_str_len = len(saving_force_str)
    saving_force_len = len(tokenizer(saving_force_str).input_ids)
    saving_force_budget = 256 if force_saving else 0
    saving_force_budget_net = saving_force_budget - saving_force_len

    step = 0
    while not all(done):
        step += 1
        active_idx = [i for i, d in enumerate(done) if not d and tokens_left[i] > 0]
        if len(active_idx) == 0: break
        active_prompts = [contexts[i] for i in active_idx]

        comp_ids_list, comp_text_list = _make_tool_inference_requests_batch(
            active_prompts,
            max_tokens=min(tokens_left[i] for i in active_idx),
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
            if stop_token == args.tool_tokens[1]:
                comp_text += args.tool_tokens[1]
            contexts[global_i] += comp_text
            stats[global_i]["generated_tokens"] += comp_ids_len
            tokens_left[global_i] -= comp_ids_len

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
                if stop_token == args.eos_token:
                    contexts[global_i] += args.eos_token
                    done[global_i] = True
                    continue
                elif tokens_left[global_i] <= 0:
                    if force_saving:
                        # There can be some issues due to tokenizer non-determinism (see bottom)
                        # thus we check for args.max_completion_length + saving_force_len even though it has
                        # args.max_completion_length + saving_force_budget_net of budget to be more conservative
                        if len(tokenizer(contexts[global_i][prompts_len[global_i]:]).input_ids) < args.max_completion_length + saving_force_len:
                            # If it already used its last saving, give it the rest of the seq length
                            if stats[global_i]["num_saving"] == max_saving:
                                tokens_left[global_i] += saving_force_budget
                                continue
                            # Force saving tool call
                            contexts[global_i] += saving_force_str
                            tokens_left[global_i] += saving_force_budget_net
                            stats[global_i]["generated_tokens"] += saving_force_len
                            continue
                        else:
                            # In case it ran out of context while calling the tool
                            detect = _detect_tool_use(contexts[global_i], prompt_idx=prompts_len[global_i], force_end_tag=False)
                            if not(detect["tool_use"] and detect["tool_type"] in ["saving", "pass"]):
                                continue
                            else:
                                print("Continuing with broken saving tool call.")
                    elif save_last_k is not None:
                        # Here it is fine to end directly as it has used the full sequence; no extra budget
                        if stats[global_i]["num_saving"] >= max_saving:
                            done[global_i] = True
                            continue
                        stats[global_i]["num_saving"] += 1

                        saving_completions_per_prompt[global_i].append(contexts[global_i][prompts_len[global_i]:] + args.eos_token)
                        saving_prompt_per_prompt[global_i].append(contexts[global_i][:prompts_len[global_i]])
                        add = args.thinking_tokens[0] + "\n" + contexts[global_i][-save_last_k:]
                        contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add

                        tokens_left[global_i] = args.max_completion_length - len(tokenizer(add).input_ids)
                        continue
                    else:
                        done[global_i] = True
                        continue
                else:
                    continue
            tool_type = detect["tool_type"]

            if tool_type in ["saving", "pass"]:
                if stats[global_i]["num_saving"] >= max_saving:
                    done[global_i] = True
                    continue
                stats[global_i]["num_saving"] += 1

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
                # elif saving_prompt == "2":
                #     add = f"\n\nBuild on this context from your last attempt:\n{summary}<|im_end|>\n<|im_start|>assistant\n{args.thinking_tokens[0]}"
                #     contexts[global_i] = contexts[global_i][:prompts_len[global_i] - len("<|im_end|>\n<|im_start|>assistant\n")] + add
                elif saving_prompt == "context":
                    add = f"\n\nContext:\n{summary}<|im_end|>\n<|im_start|>assistant\n{args.thinking_tokens[0]}"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i] - len("<|im_end|>\n<|im_start|>assistant\n")] + add
                elif saving_prompt == "special_before_thinking":
                    add = f"{args.saving_tokens[0]}\n{summary}\n{args.saving_tokens[1]}\n{args.thinking_tokens[0]}\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add
                elif saving_prompt == "special_after_thinking":
                    add = f"{args.thinking_tokens[0]}\n{args.saving_tokens[0]}\n{summary}\n{args.saving_tokens[1]}\n"
                    contexts[global_i] = contexts[global_i][:prompts_len[global_i]] + add

                tokens_left[global_i] = args.max_completion_length - len(tokenizer(add).input_ids)
            else:
                # TODO: Skip here if unknown tool type
                tool_jobs.append((global_i, deepcopy(detect), contexts[global_i], args, tokenizer))

        if tool_jobs:
            tool_start = time.time()
            with ProcessPoolExecutor(max_workers=min(len(tool_jobs), os.cpu_count())) as executor:
                results = list(executor.map(_execute_tool_job, tool_jobs))

            for global_i, tool_type, updated_context, exec_error in results:
                toks = tokenizer(updated_context).input_ids
                n_toks = len(toks)
                if n_toks >= args.max_completion_length:
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
            print(f"[Time] {tool_duration:.2f} seconds for {len(tool_jobs)} parallel tool calls. Avg {tool_duration / len(tool_jobs):.2f} seconds.")

    prompts_to_return: List[str] = []
    completions_to_return: List[str] = []

    for i in range(N):
        prompts_to_return.extend(saving_prompt_per_prompt[i])
        completions_to_return.extend(saving_completions_per_prompt[i])
        
    prompts_to_return.extend(prompts)
    for i, context in enumerate(contexts):
        completions_to_return.append(context[prompts_len[i]:])

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
        toks_len = len(toks)
        if toks_len > args.max_completion_length + saving_force_budget:
            completions_to_return[i] = tokenizer.decode(toks[:args.max_completion_length + saving_force_budget])
            toks_len_new = len(tokenizer(completions_to_return[i]).input_ids)
            print(f"[WARNING] Truncated completion {i} to fit max length: {toks_len} -> {toks_len_new}")

    print("Saving stats:", [stats[i]["num_saving"] for i in range(N)])
    print("Saving stats Avg:", sum(stats[i]["num_saving"] for i in range(N)) / N)
    print("Max Saving %:", sum(stats[i]["num_saving"] == max_saving for i in range(N)) / N * 100)
    print("Generated tokens:", [stats[i]["generated_tokens"] for i in range(N)])
    print("Generated tokens Avg:", sum(stats[i]["generated_tokens"] for i in range(N)) / N)
    print("Steps total:", step)
    return prompts_to_return, completions_to_return, stats
