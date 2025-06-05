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

"""

from __future__ import annotations

import json
import time
import requests

from typing import Dict, List, Optional, Tuple, Union
from transformers import PreTrainedTokenizerBase

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
# Core driver
################################################################################


def generate_with_tool_batch(
    prompts: List[str],
    args: GRPOConfig,
    llm: LLM = None,
    tokenizer: PreTrainedTokenizerBase = None,
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
    contexts: List[str] = prompts[:]  # running transcripts
    prompts_len = [len(prompt) for prompt in prompts]
    completions_to_return: List[str] = [] # adding intermediate savings
    prompts_to_return: List[str] = []
    tokens_left = [args.max_completion_length] * N
    done = [False] * N

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
        if step >= 20:
            break
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

        # Walk over every *active* prompt
        for local_i, global_i in enumerate(active_idx):
            # ------------------------------------------------------------------
            # 1. Book-keeping
            # ------------------------------------------------------------------
            comp_ids = comp_ids_list[local_i]
            comp_text = comp_text_list[local_i]

            tokens_left[global_i] -= len(comp_ids)
            stats[global_i]["generated_tokens"] += len(comp_ids)
            last_tok = tokenizer.decode(comp_ids[-1])

            if last_tok == args.eos_token:
                comp_text += args.eos_token
                # Still need to let it detect tool use as in case of saving tool it could end with eos

            prompt_idx = len(contexts[global_i])
            contexts[global_i] += comp_text
            # ------------------------------------------------------------------
            # 2. Look for tool calls
            # ------------------------------------------------------------------
            detect = _detect_tool_use(contexts[global_i], prompt_idx=prompt_idx)

            # ------------------------------------------------------------------
            # 3. Termination: error tool call OR no tool call OR ran out of budget
            # ------------------------------------------------------------------
            if not detect["tool_use"]:
                # Token exceeded limit / finished naturally via eos
                if (tokens_left[global_i] <= 0) or (last_tok == args.eos_token):
                    done[global_i] = True
                # Else prompt may have stopped due to min tok limit/unknown issue & needs more generation
                continue
            elif detect["error_tool_use"]:
                print(f"[{global_i}] malformed tool call – skipping tool use.")
                add = f"\n{args.result_tokens[0]}\n{args.result_tokens[1]}\n"
                n_toks = len(tokenizer(add).input_ids)
                tokens_left[global_i] -= n_toks
                stats[global_i]["generated_tokens"] += n_toks
                if tokens_left[global_i] <= 0:
                    done[global_i] = True
                    stats[global_i]["generated_tokens"] -= n_toks
                    add = ""
                contexts[global_i] += add
                continue

            # ------------------------------------------------------------------
            # 4. Execute tool & insert result
            # ------------------------------------------------------------------
            tool_type = detect["tool_type"]
            tool_start = time.time()

            if tool_type == "python":
                stats[global_i]["num_python"] += 1
                code = remove_python_prefix(text=detect["tool_argument"]['code'])
                try:
                    new_code, exec_out = python_interpreter(code)
                except Exception as e:
                    print(f"[ERROR] Python execution failed for code:\n{code}")
                    print(e)
                    exec_out = ""
                    new_code = detect["tool_argument"]['code']

                python_start_idx = detect["tool_start_idx"]
                new_python_code = json.dumps(
                    {
                        "name": "python",
                        "arguments": {
                            "code": new_code
                        }
                    }
                )

                new_text = (
                    contexts[global_i][:prompt_idx + python_start_idx]
                    + f"{args.tool_tokens[0]}\n" + new_python_code + f"\n{args.tool_tokens[1]}\n"
                    f"{args.result_tokens[0]}\n" + exec_out + f"\n{args.result_tokens[1]}\n"
                )
                toks = tokenizer(new_text[prompt_idx:]).input_ids
                n_toks = len(toks)
                tokens_left[global_i] = args.max_completion_length - n_toks
                stats[global_i]["generated_tokens"] = n_toks
                if tokens_left[global_i] <= 0:
                    done[global_i] = True
                    contexts[global_i] = tokenizer.decode(toks[:tokens_left[global_i]])
                    tokens_left[global_i] = 0
                    stats[global_i]["generated_tokens"] = args.max_completion_length
                    continue

                contexts[global_i] = new_text

            elif tool_type in ("retrieval", "retrieve"):
                stats[global_i]["num_retrieval"] += 1
                try:
                    print(detect["tool_argument"])
                    result = google_retriever(**detect["tool_argument"])
                    result_text = result.response if result else ""
                except Exception as e:
                    print(f"[ERROR] Retrieval failed.")
                    print(e)
                    result_text = ""

                new_text = f"{args.result_tokens[0]}\n" + result_text + f"\n{args.result_tokens[1]}\n"
                toks = tokenizer(new_text).input_ids
                n_toks = len(toks)
                tokens_left[global_i] -= n_toks
                stats[global_i]["generated_tokens"] += n_toks
                if tokens_left[global_i] <= 0:
                    done[global_i] = True
                    contexts[global_i] += tokenizer.decode(toks[:tokens_left[global_i]])
                    tokens_left[global_i] = 0
                    stats[global_i]["generated_tokens"] = args.max_completion_length
                    continue
                contexts[global_i] += new_text

            elif tool_type == "saving":
                stats[global_i]["num_saving"] += 1
                try:
                    summary = detect["tool_argument"]["summary"]
                except Exception as e:
                    print(e)
                    summary = ""

                # bookkeep the old thinking & summary
                completions_to_return.append(contexts[global_i][prompts_len[global_i]:])
                prompts_to_return.append(contexts[global_i][:prompts_len[global_i]])

                # Replace all old thinking with the summary
                summary = f"{args.saving_tokens[0]}\n" + summary + f"\n{args.saving_tokens[1]}\n"

                # reset tokens_left
                toks = tokenizer(summary).input_ids
                tokens_left[global_i] = args.max_completion_length - len(toks)
            
                if tokens_left[global_i] <= 0:
                    done[global_i] = True
                    summary = tokenizer.decode(toks[:tokens_left[global_i]])
                    tokens_left[global_i] = 0

                contexts[global_i] = (
                    contexts[global_i][: prompts_len[global_i]] # has \n at the end
                    + summary # TODO: Check if needs <think> at the end
                )
                prompts[global_i] = contexts[global_i]
                prompts_len[global_i] = len(contexts[global_i])

            tool_duration = time.time() - tool_start
            print(f"[Time] {tool_type} tool call took {tool_duration:.2f} seconds")

    # one more short call to get the final answer
    # _, final_answer = _make_tool_inference_requests_batch(
    #     contexts,
    #     max_tokens=128,
    #     args=args,
    #     stop=[args.eos_token],
    #     llm=llm,
    # )
    # for i in range(len(contexts)):
    #     contexts[i] += final_answer[i] + f"\n{args.eos_token}"

    prompts_to_return.extend(prompts)
    for i, context in enumerate(contexts):
        completions_to_return.append(context[prompts_len[i]:])
    
    # **IMPORTANT**: The returns are formatted as follows. Suppose there's 3 prompts, and each has 3 saving segments.
    # output0a looks like “thinking, saving”, output1b looks like “saving 1, thinking 2, saving 2", output1c looks like “saving 2, thinking 3, answer”
    # prompts_to_return = [prompts[0], prompts[0], ..., prompts[2], prompts[2], prompts[0], ..., prompts[2]]
    # completions_to_return = [output0a, output0b, ..., output2a, output2b, output0c, ..., output2c]
    return prompts_to_return, completions_to_return, stats
    
    # to test RL masking, comment the above return and uncomment the following
    # return [prompts[0], prompts[1]], [example_output1c, example_output2c], stats
    
    # to test inftythink reward broadcasting, comment the above return and uncomment the following

    # return [prompts[0], prompts[0], prompts[1], prompts[1], prompts[0], prompts[1]], [example_output1a, example_output1b, example_output2a, example_output2b, example_output1c, example_output2c], stats

example_output1a = """
<think>
Let me attempt a diagram and use triangle properties to compute AD². Since ABCD is cyclic, and ∠DBC is 15°, this may involve using extended Law of Sines or triangle similarity. I’ll first sketch the geometry to better visualize the problem.
</think>
<tool_call>
{"name": "saving", "arguments": {"summary": "Draw diagram and recall triangle properties to prepare for using extended Law of Sines or triangle similarity."}}
</tool_call>
<|im_end|>
"""

example_output1b = """
<saving>
Draw diagram and recall triangle properties to prepare for using extended Law of Sines or triangle similarity.
</saving>
<think>
From the diagram, I can label the triangle ΔBDC and apply the Extended Law of Sines:  
For triangle BDC, with R = 100√3 and ∠DBC = 15°, use:
\[ \frac{BC}{\sin(\angle BDC)} = 2R \]
We can reverse compute BC, then use triangle relationships to find AD.
</think>
<tool_call>
{"name": "saving", "arguments": {"summary": "Apply extended Law of Sines on triangle BDC with R=100√3 and ∠DBC=15° to compute BC."}}
</tool_call>
<|im_end|>
"""


example_output1c = """
<saving>
Draw diagram and recall triangle properties to prepare for using extended Law of Sines or triangle similarity.
</saving>
<think>
Let me attempt a diagram and use triangle properties to compute AD². Since ABCD is cyclic, and ∠DBC is 15°, this may involve using extended Law of Sines or triangle similarity. I’ll first sketch the geometry to better visualize the problem.
</think>
<tool_call>
{"name": "plot_geometry_diagram", "arguments": {"type": "cyclic_quadrilateral", "diagonal": 300, "angle_DBC_deg": 15, "radius": 100.0 * 3 ** 0.5}}
</tool_call>
<tool_response>
{"image_url": "https://dummyimages.com/cyclic_quadrilateral_annotated.png"}
</tool_response>
<think>
From the diagram, I can label the triangle ΔBDC and apply the Extended Law of Sines:  
For triangle BDC, with R = 100√3 and ∠DBC = 15°, use:
\[ \\frac{BC}{\sin(\\angle BDC)} = 2R \]
We can reverse compute BC, then use triangle relationships to find AD.
</think>
<tool_call>
{"name": "compute_extended_law_of_sines", "arguments": {"R": 100 * (3)**0.5, "angle_deg": 15}}
</tool_call>
<tool_response>
{"result": 200.0}
</tool_response>
<think>
So BC = 200. Let’s now use the Cosine Law in triangle ADC using known sides and angles.
</think>
<tool_call>
{"name": "cosine_law_solver", "arguments": {"side1": 300, "side2": 200, "angle_deg": 120}}
</tool_call>
<tool_response>
{"result": 519.6152422706632}
</tool_response>
**Final Answer**
\\boxed{60000}

**Step-by-Step Explanation:**
1. Drew a diagram of the cyclic quadrilateral.
2. Used Extended Law of Sines in ΔBDC to find BC = 200.
3. Applied Cosine Law to triangle ADC.
4. Found that AD ≈ 519.6, hence AD² ≈ 270000.
<|im_end|>
"""


example_output2a = """
<think>
To solve for the maximum height, we need the vertical component of velocity and apply the formula:
\[ h = \\frac{v_y^2}{2g} \]
Let’s compute the vertical component first.
</think>
<tool_call>
{"name": "saving", "arguments": {"summary": "Plan to compute vertical velocity component and apply projectile height formula."}}
</tool_call>
<|im_end|>
"""

example_output2b = """
<saving>
Computed vertical velocity v_y = 30 m/s using angle decomposition.
</saving>
<think>
Vertical velocity is 30 m/s. Now calculate the maximum height.
</think>
<tool_call>
{"name": "saving", "arguments": {"summary": "Use formula h = v_y² / (2g) with v_y = 30 and g = 9.8 to compute height."}}
</tool_call>
<|im_end|>
"""

example_output2c = """
<saving>
Computed vertical velocity v_y = 30 m/s using angle decomposition.
</saving>
<think>
To solve for the maximum height, we need the vertical component of velocity and apply the formula:
\[ h = \\frac{v_y^2}{2g} \]
Let’s compute the vertical component first.
</think>
<tool_call>
{"name": "decompose_velocity", "arguments": {"velocity": 60, "angle_deg": 30}}
</tool_call>
<tool_response>
{"vx": 51.9615, "vy": 30.0}
</tool_response>
<think>
Vertical velocity is 30 m/s. Now calculate the maximum height.
</think>
<tool_call>
{"name": "compute_max_height", "arguments": {"vy": 30.0, "g": 9.8}}
</tool_call>
<tool_response>
{"result": 45.91836734693877}
</tool_response>
<think>
The maximum height is approximately 45.92 meters. I’ll cross-verify this with a physics knowledge source.
</think>
<tool_call>
{"name": "google_retriever", "arguments": {"query": "maximum height of projectile formula physics"}}
</tool_call>
<tool_response>
{"summary": "The maximum height of a projectile is given by h = (v_y^2) / (2g), where v_y is the initial vertical velocity and g is the acceleration due to gravity."}
</tool_response>
**Final Answer**
\\boxed{45.92\ \\text{m}}

**Step-by-Step:**
1. Vertical component: \(v_y = 60 \sin(30^\circ) = 30\ \\text{m/s}\)
2. Applied height formula: \(h = \\frac{30^2}{2 \cdot 9.8} ≈ 45.92\ \\text{m}\)
3. Verified with external source.
<|im_end|>
"""