import json
import requests
import time
from tqdm import tqdm
import trl
import multiprocessing
import re

from typing import Any, Dict, List, Optional, Union, Tuple

from .python_and_retriever import google_retriever,python_interpreter, remove_python_prefix
from ..trainer.grpo_config import GRPOConfig


def _make_tool_inference_requests(
    prompt: str,
    max_tokens: int,
    args: trl.GRPOConfig,
    stop: Optional[Union[str, List[str]]] = None,
) -> Tuple[List[List[int]], List[str]]:
    session = requests.Session()
    url = f"http://{args.vllm_server_host}:{args.vllm_server_port}/generate/"
    prompts = [prompt]
    response = session.post(
        url,
        json={
            "prompts": prompts,
            "n": 1, # tool inference requires each process to handle one thinking trajectory
            "repetition_penalty": args.repetition_penalty,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": -1 if args.top_k is None else args.top_k,
            "min_p": 0.0 if args.min_p is None else args.min_p,
            "max_tokens": max_tokens,
            "guided_decoding_regex": args.vllm_guided_decoding_regex,
            "stop": stop,
        },
    )
    if response.status_code == 200:
        return response.json()["completion_ids"], response.json()["completion_text"]
    else:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")


def _detect_tool_use(context: str, prompt_idx: int) -> dict:
    """
    Detect tool usage in a generation context. Extracts:
    - Python code, Retrieval query or Saved summary
    - between <tool_call> and </tool_call>
    
    Returns a dict:
    {
        "tool_use": bool,
        "tool_type": "python" | "retrieval" | "saving" | None,
        "tool_argument": str or (start_idx, end_idx, extracted_string) or None,
        "error_tool_use": bool
    }
    """
    result = {
        "tool_use": False,
        "tool_type": None,
        "tool_argument": None,
        "error_tool_use": False
    }

    generated = context[prompt_idx:]

    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    response_tag = "</tool_response>"
    
    # Find the last occurrence of tool calling
    start = generated.rfind(start_tag)
    end = generated.rfind(end_tag, start)
    
    # Find the last occurrence of tool result
    last_response_idx = generated.rfind(response_tag, start)
    
    # no tool call detected
    if start == -1:
        return result
    
    # last tool call is processed
    if start != -1 and last_response_idx != -1:
        return result
    
    # If </tool_call> is not at the end of the generation because of vllm stop
    # then end = -1
    
    if end == -1:
        tool_call_raw = generated[start + len(start_tag):].strip()
    else:
        tool_call_raw = generated[start + len(start_tag):end].strip()
    
    try:            
        tool_call_raw = tool_call_raw.encode('utf-8').decode('unicode_escape').strip()
        tool_call_raw = re.sub(r'(?<!\\)\n', r'\\n', tool_call_raw)
        tool_call_raw = tool_call_raw.replace('“', '\\"').replace('”', '\\"')
        
        parsed = json.loads(tool_call_raw)
        result = {
            "tool_use": True,
            "tool_type": parsed["name"],
            "tool_argument": parsed["arguments"],
            "error_tool_use": False,
        }
        return result
    except json.JSONDecodeError as e:
        print(e)
    
    # if json parsing fails, fall back to regex
    code_match = re.search(r'"code"\s*:\s*"(.+?)"\s*}', tool_call_raw, re.DOTALL)
    query_match = re.search(r'"query"\s*:\s*"(.+?)"\s*}', tool_call_raw, re.DOTALL)
    print("Using regex to capture tool call.")
    if code_match:
        result = {
            "tool_use": True,
            "tool_type": "python",
            "tool_argument": {"code": code_match.group(1)},
            "error_tool_use": False,
        }
    elif query_match:
        result = {
            "tool_use": True,
            "tool_type": "retrieval",
            "tool_argument": {"query": query_match.group(1)},
            "error_tool_use": False,
        }
    else:
        result = {
            "tool_use": True,
            "tool_type": None,
            "tool_argument": None,
            "error_tool_use": True,
        }
    print(result)
    return result

def _generate_with_tool(
    question: str, args: GRPOConfig
) -> str:
    """
    Generates text from the LLM using vLLM, detects tool triggers,
    invokes tools (python or retrieval), and updates the context accordingly.
    """    
    # Initialize the context, we don't do manual formatting anymore
    context = question
    # context = SYSTEM_PROMPT + question + f"\n<|im_end|>\n{args.assistant_token}\n{args.thinking_tokens[0]}\n"
    
    tokens_left = args.max_completion_length
    # TODO: maybe change prompt_ids with inftythink
    prompt_idx = len(context)
    start_total_time = time.time()
    
    # we capture all summarized reasoning in return_ids
    # in the case of 2 summaries,
    # return_ids would be [
    #   [thinking1, summary1]
    #   [summary1, thinking2, summary2]
    #   [summary2, thinking3, answer]
    # ]
    # and update them with the same reward of the final pass
    return_ids = []
    
    tool_use_stat = {
        "tool_use": False,
        "num_python": 0,
        "num_retrieval": 0,
        "num_saving": 0,
    }

    
    while True:
        gen_start = time.time()
        # TODO: change this to streaming
        completion_ids, completion_text = _make_tool_inference_requests(
                prompt=context,
                max_tokens=tokens_left,
                args=args,
                stop=[args.tool_tokens[0], args.eos_token],
            )
        completion_ids = completion_ids[0]
        completion_text = completion_text[0]
        print(completion_text)
        gen_duration = time.time() - gen_start
        print(f"[Time] LLM generation took {gen_duration:.2f} seconds.")
        # Append the new output to the context
        context = context + completion_text
        print(len(completion_ids))
        tokens_left -= len(completion_ids)
        
        detect_result = _detect_tool_use(context, prompt_idx, tokens_left, args)
        if detect_result["error_tool_use"]:
            print("[ERROR] Malformed tool call detected.")

            
        if detect_result["tool_use"] == False or tokens_left == 0:
            context += "\n<|im_start|>answer\n"
            return_ids, completion_text = _make_tool_inference_requests(
                prompt=context,
                max_tokens=512,
                args=args,
                stop=[args.eos_token]
            )
            context += f"\n{completion_text}\n{args.eos_token}"
            break
        
        
        # After the previous if argument, now the stop reason must be getting a result token.
        context += f'\n{args.result_token}\n'
        tool_use_stat["tool_use"] = True
        tool_type = detect_result["tool_type"]

        if tool_type == "python":
            code = remove_python_prefix(**detect_result["tool_argument"])
            tool_start = time.time()
            try:
                new_code, execution_result = python_interpreter(code)
            except Exception as e:
                import traceback
                print(f"[ERROR] Python execution failed for question:\n{question}")
                traceback.print_exc()
                new_code, execution_result = None, ""

            tool_duration = time.time() - tool_start
            print(f"[Time] Python tool call took {tool_duration:.2f} seconds")

            # inserting the result between <tool_response> and </tool_response>
            context = context + "\n<tool_response>\n" + execution_result + "\n</tool_response>\n"


        elif tool_type == "retrieval" or tool_type == "retrieve":
            query = detect_result["tool_argument"]
            tool_start = time.time()
            try:
                print(f"Query for Google retrieval: '{query['query']}'")
                execution_result = google_retriever(**query)
                result_text = execution_result.response if execution_result else ""
            except Exception as e:
                import traceback
                print(f"[ERROR] Retrieval failed.")
                traceback.print_exc()
                result_text = ""

            tool_duration = time.time() - tool_start
            print(f"[Time] Retrieval tool call took {tool_duration:.2f} seconds")

            # TODO: fix this
            context = context + result_text + "\n<|im_start|>continue\n"

        elif tool_type == "saving":
            # replace previous thinking with summary in the tool call argument
            extracted = detect_result["tool_argument"]
            print(f"[INFO] Detected saving block: {extracted}")
            context = context[:prompt_idx] + f"\n{args.saving_tokens[0]}\n" + extracted + f"\n{args.saving_tokens[1]}\n"
            

    total_duration = time.time() - start_total_time
    print(f"[Time] Total processing time: {total_duration:.2f} seconds")
    print(f"[INFO] Generated {args.max_completion_length - tokens_left} tokens")
    print(tool_use_stat)
    
    
    # return return_ids[0], tool_use_stat
    return context, tool_use_stat

def _generate_with_tool_safe(input_argument: dict) -> str:
    
    question = input_argument['question']
    args = input_argument['args']
    try:
        return question, _generate_with_tool(question, args)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception for question:\n{question}\n")
        traceback.print_exc()
        return f"[ERROR] {str(e)}", False



def generate_with_tool_async(questions: list, args: GRPOConfig, max_processes: int = 20) -> list:
    """
    Batch inference - Parallel version using multiprocessing.
    Maintains a pool of `max_processes` workers and processes questions as workers free up.
    """
    bar = tqdm(total=len(questions))
    results = []
    
    input_arguments = [
        {
            "question": questions[i],
            "args": args,
        } for i in range(len(questions))
    ]
    
    with multiprocessing.Pool(processes=max_processes) as pool:
        for question, output, tool_stats in pool.imap(_generate_with_tool_safe, input_arguments):
            results.append({
                "system_prompt": args.system_prompt,
                "question": question,
                "output": output,
                "tool_stats": tool_stats
            })
            bar.update(1)
            
    return results



if __name__ == "__main__":
    example = r'''<tool_call>\n{"name": "python", "arguments": {"code": "navin_hourly_rate_usd = 160 / 32.35\nluka_hourly_rate_usd = 25 / 5.18\nian_hourly_rate_usd = 34 / 6.95\nnavin_total = navin_hourly_rate_usd * 8\nluka_total = luka_hourly_rate_usd * 8\nian_total = ian_hourly_rate_usd * 8\nprint("Navin's total: ", navin_total)\nprint("Luka's total: ", luka_total)\nprint("Ian's total: ",ian_total)"}}</tool_call>\n'''
    detect = _detect_tool_use(
        example,
        prompt_idx=0,
    )
    print(detect)