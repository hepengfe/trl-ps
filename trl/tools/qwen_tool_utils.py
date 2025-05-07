import json
import requests
import time
from tqdm import tqdm
import trl

from typing import Any, Dict, List, Optional, Set, Union

from .python_and_retriever import google_retriever,python_interpreter, remove_python_prefix
from ..trainer.grpo_config import GRPOConfig



SYSTEM_PROMPT = """<|im_start|>system
You are a helpful assistant. You are good at following instructions and using given tools.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "execute_bash", "description": "THIS IS A LONG TEXT ABOUT THIS FUNCTION", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute."}, "is_input": {"type": "string", "description": "If True, the command is an input to the running process. If False, the command is a bash command to be executed in the terminal. Default is False.", "enum": ["true", "false"]}}, "required": ["command"]}}}
</tools>
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>


Role Description:
You are an intelligent assistant whose job is to solve problems using a rigorous, multi-step reasoning process. To guarantee both precision and factual accuracy, you are **required to use external tools**—specifically, the web retrieval tool and the Python interpreter—for every non-trivial computation or fact verification step. You must never perform such steps solely in plain text; instead, you must always use the exact tool call format provided below.

Task Requirements:
1. Comprehensive and Iterative Reasoning:
- Conduct a full analysis including summarization, reassessment, and reflection.
- **Force Tool Invocation:** For every non-trivial step (e.g., verifying facts, performing arithmetic, or confirming data), immediately and mandatorily use the corresponding tool call using the formats given below. Do not simply describe the process in text—insert a tool invocation using the specified syntax.

2. Mandated Tool Usage (No Exceptions):
- **Web Retrieval Tool:**
<tools>
{"type":"function","function":{"name":"retrieval","description":"Used to verify factual information or retrieve up-to-date external content via a search engine. This tool ensures factual accuracy by accessing web-based data sources.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search query or question to be retrieved from the web."}},"required":["query"]}}}
</tools>

When you need to verify any factual information or retrieve current data, you MUST insert the following command immediately at the point of need:

<tool_call>
{"name": "retrieval", "arguments": {"query": <replace_with_your_retrieval_query>}}
</tool_call>


- **Python Interpreter Tool:**
<tools>
{"type":"function","function":{"name":"python","description":"Used for performing calculations, debugging code, or validating intermediate results through Python execution. Ensures correctness for all non-trivial quantitative or logical steps.","parameters":{"type":"object","properties":{"code":{"type":"string","description":"A concise Python code snippet to execute. Must be self-contained and return the desired output."}},"required":["code"]}}}
</tools>

For any calculations, code debugging, or intermediate quantitative validations, you MUST insert a single, concise code snippet using the format below:

<tool_call>
{"name": "python", "arguments": {"code": <replace_with_your_python_code>}}
</tool_call>


- **Thinking Summarization Tool:**
For any calculations, code debugging, or intermediate quantitative validations, you MUST insert a single, concise code snippet using the format below:
If the previous thinking is too lengthy, you MUST call Thinking Summarization Tool to replace previous thinking with summarization.

<tool_call>
{"name": "saving", "arguments": None}
</tool_call>


**Important Guidelines:**
- Every time you perform a non-trivial calculation or need to verify a fact, you must call the appropriate tool using the exact syntax provided. No calculation or verification step should be described purely in plain text.
- Do not duplicate the tool call; only one invocation per verification step is allowed.
- All tool invocations must strictly follow these formats—any deviation is not permitted.

3. Building on Partial Answers:
- If a partial answer or previous reasoning is provided, continue from that content. Do not start from scratch; integrate additional tool calls as needed to complete and refine the solution.

4. Final Answer Format:
- End your response with the final answer using the precise format below:


**Final Answer**


\\boxed{Your Answer}


- The final answer must be enclosed within LaTeX using `\\boxed{}` immediately after the **Final Answer** header.

Key Reminder:
From now on, **every** non-trivial computation, fact verification, or any step requiring external validation must be performed using the designated tool calls in the exact format provided. This approach is mandatory for ensuring that your entire chain-of-thought, including intermediate conclusions, is fully backed by executable computations and reliable external references.

Respond in the following format:
<think>
reasoning process here
</think>
answer here

<|im_end|>
<|im_start|>user
"""

def _make_tool_inference_requests(
    prompt: str,
    max_tokens: int,
    args: trl.GRPOConfig,
    stop: Optional[Union[str, List[str]]] = None,
) -> list[list[str]]:
    session = requests.Session()
    url = f"http://{args.vllm_server_host}:{args.vllm_server_port}/generate/"
    response = session.post(
        url,
        json={
            "prompt": prompt,
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


def _detect_tool_use(context: str, prompt_idx: int, tokens_left: int, args: GRPOConfig) -> dict:
    """
    Detect tool usage in a generation context. Extracts:
    - Python code between <|im_start|>python and <|im_start|>result
    - Retrieval query between <|im_start|>retrieval and <|im_start|>result
    - Saved summary between last <save> and </save> (with error detection for nesting)
    
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
    
    # TODO: tool call is not at the end of the generation, ignore?
    
    try:
        tool_call_raw = generated[start + len(start_tag):end].strip()
        parsed = json.loads(tool_call_raw)
        result = {
            "tool_use": True,
            "tool_type": parsed["name"],
            "tool_argument": parsed["arguments"],
            "error_tool_use": True
        }
    except json.JSONDecodeError:
        result = {
            "tool_use": True,
            "tool_type": None,
            "tool_argument": None,
            "error_tool_use": True
        }

    return result

def _generate_with_tool_safe(input_argument: dict) -> str:
    
    def _generate_with_tool(
        question: str, args: GRPOConfig
    ) -> str:
        """
        Generates text from the LLM using vLLM, detects tool triggers,
        invokes tools (python or retrieval), and updates the context accordingly.
        """    
        # Initialize the context
        context = SYSTEM_PROMPT + question + f"\n{args.assistant_token}\n{args.thinking_tokens[0]}\n"
        
        tokens_left = args.max_completion_length
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
            completion_ids, completion_text = _make_tool_inference_requests(
                    prompt=context,
                    max_tokens=tokens_left,
                    args=args,
                    stop=[args.result_token, args.eos_token, args.saving_token],
                )
            completion_ids, completion_text = completion_ids[0], completion_text[0]
            gen_duration = time.time() - gen_start
            print(f"[Time] LLM generation took {gen_duration:.2f} seconds.")
            # Append the new output to the context
            context = context + completion_text
            tokens_left -= len(completion_ids)
            
            detect_result = _detect_tool_use(context, prompt_idx, tokens_left, args)
            if detect_result["error_tool_use"]:
                print("[ERROR] Malformed tool call detected.")
                
                
            if detect_result["tool_type"] is None or tokens_left == 0:
                context += "\n<|im_start|>answer\n"
                return_ids, completion_text = _make_tool_inference_requests(
                    prompt=context,
                    max_tokens=512,
                    args=args,
                    stop=[args.eos_token]
                )
                context += f"\n{args.eos_token}"
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

                last_tool_index = context.rfind(args.python_token)
                if new_code:
                    context = context[:last_tool_index] + new_code + "\n<|im_start|>result\n" + execution_result + "\n<|im_start|>continue\n"
                else:
                    context = context[:last_tool_index] + "\n<|im_start|>result\n" + execution_result + "\n<|im_start|>continue\n"

            elif tool_type == "retrieval":
                query = detect_result["tool_argument"]
                tool_start = time.time()
                try:
                    print(f"Query for Google retrieval: '{query["query"]}'")
                    execution_result = google_retriever(**query)
                    result_text = execution_result.response if execution_result else ""
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Retrieval failed.")
                    traceback.print_exc()
                    result_text = ""

                tool_duration = time.time() - tool_start
                print(f"[Time] Retrieval tool call took {tool_duration:.2f} seconds")

                result_index = context.rfind(args.result_token)
                context = context[:result_index + len(args.result_token)] + result_text + "\n<|im_start|>continue\n"

            elif tool_type == "saving":
                start_idx, end_idx, extracted = detect_result["tool_argument"]
                print(f"[INFO] Detected saving block: {extracted}")
                context = context[:prompt_idx + start_idx] + extracted + context[prompt_idx + end_idx:]
                # Resume generation right after saving
                context += "\n<|im_start|>continue\n"
                

        total_duration = time.time() - start_total_time
        print(f"[Time] Total processing time: {total_duration:.2f} seconds")
        print(f"[INFO] Generated {args.max_completion_length - tokens_left} tokens")
        print(tool_use_stat)
        
        
        return return_ids[0], tool_use_stat
    
    
    question = input_argument['question']
    args = input_argument['args']
    try:
        return _generate_with_tool(question, args)
    except Exception as e:
        import traceback
        print(f"[ERROR] Exception for question:\n{question}\n")
        traceback.print_exc()
        return f"[ERROR] {str(e)}", False

