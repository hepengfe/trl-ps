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
    idx, detect, context, args_, tokenizer = args
    try:
        tool_type = detect["tool_type"]
        if tool_type == "python":
            if isinstance(detect["tool_argument"], str):
                code = detect["tool_argument"]
            else:
                code = detect["tool_argument"]["code"]
                
            try:
                new_code, exec_out = python_interpreter(code)
            except Exception as e:
                print(f"[ERROR] Python execution failed for idx={idx}:\n{code}\n{e}")
                exec_out = ""
                new_code = code

            new_python_code = json.dumps({
                "name": "python",
                "arguments": {"code": new_code}
            })
            insert = (
                f"{args_.tool_tokens[0]}\n{new_python_code}\n{args_.tool_tokens[1]}\n"
                f"{args_.result_tokens[0]}\n{exec_out}\n{args_.result_tokens[1]}\n"
            )
            return idx, insert, tool_type

        elif tool_type in ("retrieval", "retrieve"):
            try:
                result = google_retriever(**detect["tool_argument"])
                result_text = result.response if result else ""
            except Exception as e:
                print(f"[ERROR] Retrieval failed for idx={idx}: {e}")
                result_text = ""

            insert = (
                f"{args_.result_tokens[0]}\n{result_text}\n{args_.result_tokens[1]}\n"
            )
            return idx, insert, tool_type

    except Exception as e:
        print(f"[ERROR] Tool job failed for idx={idx}: {e}")
        return idx, "", None


################################################################################
# Core driver
################################################################################

def generate_with_tool_batch(
    prompts: List[str],
    args: GRPOConfig,
    llm: LLM = None,
    tokenizer = None,
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
    completions_to_return = []
    contexts_to_return = []
    prompts_to_return = []
    tokens_left = [args.max_completion_length] * N
    done = [False] * N
    stats = [{"generated_tokens": 0, "num_python": 0, "num_retrieval": 0, "num_saving": 0} for _ in range(N)]

    step = 0
    while not all(done) and step < 20:
        step += 1
        active_idx = [i for i, d in enumerate(done) if not d and tokens_left[i] > 0]
        if len(active_idx) == 0:
            break
        active_prompts = [contexts[i] for i in active_idx]

        comp_ids_list, comp_text_list = _make_tool_inference_requests_batch(
            active_prompts,
            max_tokens=min(tokens_left[i] for i in active_idx),
            args=args,
            stop=[args.result_tokens[0], args.eos_token],
            llm=llm,
        )

        # Update contexts and detect tool use
        tool_jobs = []
        for local_i, global_i in enumerate(active_idx):
            comp_ids = comp_ids_list[local_i]
            comp_text = comp_text_list[local_i]
            contexts[global_i] += comp_text
            stats[global_i]["generated_tokens"] += len(comp_ids)
            tokens_left[global_i] -= len(comp_ids)

            detect = _detect_tool_use(
                contexts[global_i],
                prompt_idx=len(contexts[global_i]) - len(comp_text),
            )
            if detect["error_tool_use"]:
                print(f"[{global_i}] malformed tool call – skipping tool use.")
                contexts[global_i] += f"\n{args.result_tokens[0]}\n{args.result_tokens[1]}\n"
                continue
            
            if not detect["tool_use"]:
                if tokens_left[global_i] <= 0 or "Final Answer" in contexts[global_i]:
                    done[global_i] = True
                    continue
                else:
                    continue

            tool_type = detect["tool_type"]
            if tool_type == "saving":
                stats[global_i]["num_saving"] += 1
                try:
                    summary = detect["tool_argument"]["summary"]
                except:
                    summary = ""

                contexts_to_return.append(contexts[global_i] + f"\n{args.eos_token}\n")
                completions_to_return.append(contexts[global_i][prompts_len[global_i]:] + f"\n{args.eos_token}\n")
                prompts_to_return.append(contexts[global_i][:prompts_len[global_i]])

                contexts[global_i] = (
                    contexts[global_i][:prompts_len[global_i]]
                    + f"\n{args.saving_tokens[0]}\n{summary}\n{args.saving_tokens[1]}\n"
                )
                toks = len(tokenizer(f"\n{args.saving_tokens[0]}\n" + summary + f"\n{args.saving_tokens[1]}\n").input_ids)
                tokens_left[global_i] = args.max_completion_length - toks

            else:
                tool_jobs.append((global_i, detect, contexts[global_i], args, tokenizer))

        tool_start = time.time()
        if tool_jobs:
            with mp.Pool(min(len(tool_jobs), mp.cpu_count())) as pool:
                results = pool.map(_execute_tool_job, tool_jobs)

            for global_i, insertion, tool_type in results:
                if not insertion:
                    done[global_i] = True
                    continue
                contexts[global_i] += insertion
                toks = len(tokenizer(insertion).input_ids)
                
                # stats[global_i]["generated_tokens"] -= len(comp_ids)
                # tokens_left[global_i] += len(comp_ids)
                # stats[global_i]["generated_tokens"] += toks
                # tokens_left[global_i] -= toks
                
                if tool_type == "python":
                    stats[global_i]["num_python"] += 1
                elif tool_type in ("retrieval", "retrieve"):
                    stats[global_i]["num_retrieval"] += 1
        
        tool_duration = time.time() - tool_start
        print(f"[Time] Parallel tool call took {tool_duration:.2f} seconds. Avg {tool_duration / len(tool_jobs):.2f} seconds.")


    contexts_to_return.extend(contexts)
    prompts_to_return.extend(prompts)
    for i, context in enumerate(contexts):
        completions_to_return.append(context[prompts_len[i]:])

    return prompts_to_return, completions_to_return, stats
