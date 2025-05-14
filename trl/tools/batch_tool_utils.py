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
    4.  Execute the tool(s) synchronously, insert <tool_response>…, and mark
        that prompt for another generation pass.
    5.  If a prompt finishes (no tool call or `tokens_left==0`) produce its
        final answer and mark it *done*.
* Loop until every prompt is *done*.

Assumptions
-----------
* `_detect_tool_use`, `python_interpreter`, `google_retriever`, and
  `remove_python_prefix` are kept unchanged.
* vLLM server is **stream-free**; if you stream, swap the simple POST for a
  streaming iterator.
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple, Union

import requests
import trl

from .python_and_retriever import (
    google_retriever,
    python_interpreter,
    remove_python_prefix,
)
from ..trainer.grpo_config import GRPOConfig
from .python_and_retriever import google_retriever,python_interpreter, remove_python_prefix
from .async_tool_utils import SYSTEM_PROMPT, _detect_tool_use

################################################################################
# vLLM helpers
################################################################################


def _make_tool_inference_requests_batch(
    prompts: List[str],
    max_tokens: int,
    args: GRPOConfig,
    stop: Optional[List[str]] | None = None,
) -> Tuple[List[List[int]], List[str]]:
    """
    Submit **N** prompts in one request and return (ids_list, text_list) with
    matching order.
    """
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


################################################################################
# Core driver
################################################################################


def batch_generate_with_tool(
    prompts: List[str],
    args: GRPOConfig,
) -> Tuple[List[str], List[Dict[str, int]]]:
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
    contexts: List[str] = prompts[:]  # running transcripts
    tokens_left = [args.max_completion_length] * N
    done = [False] * N

    # simple counters
    stats = [
        {
            "generated_tokens": 0,
            "num_python": 0,
            "num_retrieval": 0,
            "num_saving": 0,
        }
        for _ in range(N)
    ]

    step = 0
    while not all(done):
        step += 1
        active_idx = [i for i, d in enumerate(done) if not d]
        active_prompts = [contexts[i] for i in active_idx]

        comp_ids_list, comp_text_list = _make_tool_inference_requests_batch(
            active_prompts,
            max_tokens=max(tokens_left[i] for i in active_idx),
            args=args,
            stop=[args.tool_tokens[0], args.eos_token],
        )

        # Walk over every *active* prompt
        for local_i, global_i in enumerate(active_idx):
            # ------------------------------------------------------------------
            # 1. Book-keeping
            # ------------------------------------------------------------------
            comp_ids = comp_ids_list[local_i]
            comp_text = comp_text_list[local_i]
            contexts[global_i] += comp_text
            stats[global_i]["generated_tokens"] += len(comp_ids)
            tokens_left[global_i] -= len(comp_ids)

            # ------------------------------------------------------------------
            # 2. Look for tool calls
            # ------------------------------------------------------------------
            detect = _detect_tool_use(
                contexts[global_i],
                prompt_idx=len(contexts[global_i]) - len(comp_text),
                tokens_left=tokens_left[global_i],
                args=args,
            )

            if detect["error_tool_use"]:
                print(f"[{global_i}] malformed tool call – skipping.")
                done[global_i] = True
                continue

            # ------------------------------------------------------------------
            # 3. Termination: no tool call OR ran out of budget
            # ------------------------------------------------------------------
            if not detect["tool_use"] or tokens_left[global_i] <= 0:
                contexts[global_i] += "\n</think>\n**Final Answer**\n"
                done[global_i] = True
                continue

            # ------------------------------------------------------------------
            # 4. Execute tool & insert result
            # ------------------------------------------------------------------
            tool_type = detect["tool_type"]
            contexts[global_i] += f"\n{args.result_token}\n"

            if tool_type == "python":
                stats[global_i]["num_python"] += 1
                code = remove_python_prefix(**detect["tool_argument"])
                _, exec_out = python_interpreter(code)
                contexts[global_i] += (
                    "<tool_response>\n" + exec_out + "\n</tool_response>\n"
                )

            elif tool_type in ("retrieval", "retrieve"):
                stats[global_i]["num_retrieval"] += 1
                result = google_retriever(**detect["tool_argument"])
                result_text = result.response if result else ""
                contexts[global_i] += (
                    "<tool_response>\n" + result_text + "\n</tool_response>\n"
                )

            elif tool_type == "saving":
                stats[global_i]["num_saving"] += 1
                summary = detect["tool_argument"]
                # Replace everything since last user turn with the summary
                contexts[global_i] = (
                    contexts[global_i][: detect["tool_argument_start"]]
                    + f"\n{args.saving_tokens[0]}\n"
                    + summary
                    + f"\n{args.saving_tokens[1]}\n"
                )

            # After tool response we go for another generation pass (not done)
            
    # one more short call to get the final answer
    _, final_answer = _make_tool_inference_requests_batch(
        contexts,
        max_tokens=512,
        args=args,
        stop=[args.eos_token],
    )
    for i in range(len(contexts)):
        contexts[i] += final_answer[i] + f"\n{args.eos_token}"

    return contexts, stats
