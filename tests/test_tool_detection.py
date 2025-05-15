import json
from datasets import load_dataset
from trl.trainer.grpo_config import GRPOConfig
from trl.tools.async_tool_utils import generate_with_tool_async, _generate_with_tool_safe
from trl.tools.batch_tool_utils import generate_with_tool_batch
import random
from tqdm import tqdm
import multiprocessing

SYSTEM_PROMPT =  f"""
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{"type":"function","function":{{"name":"retrieval","description":"Used to verify factual information or retrieve up-to-date external content via a search engine. This tool ensures factual accuracy by accessing web-based data sources.","parameters":{{"type":"object","properties":{{"query":{{"type":"string","description":"The search query or question to be retrieved from the web."}}}},"required":["query"]}}}}}}
{{"type":"function","function":{{"name":"python","description":"Used for performing calculations, debugging code, or validating intermediate results through Python execution. Ensures correctness for all non-trivial quantitative or logical steps.","parameters":{{"type":"object","properties":{{"code":{{"type":"string","description":"A concise Python code snippet to execute. Must be self-contained and return the desired output."}}}},"required":["code"]}}}}}}
{{"type":"function","function":{{"name":"saving","description":"Used to replace previous reasoning with a concise summary when the prior thinking is too lengthy. This tool ensures the assistant remains focused and avoids unnecessary verbosity.","parameters":{{"type":"object","properties":{{"summary":{{"type":"string","description":"Summarization of all previous thinking trajectory"}}}},"required":["summary"]}}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

- The final answer must be enclosed within LaTeX using `\\boxed{{}}` immediately after the **Final Answer** header.
"""


output_path = "/workspace/s1-tool/s1_tool/tools/jsonl/qwen3_output_5_13.json"


# Load a small number of test samples
# dataset = load_dataset("simplescaling/aime24_nofigures", split="train[:3]")  # only 3 for quick test
dataset = load_dataset("simplescaling/aime24_nofigures", split="train")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)


# Prepare input prompts
prompt_prefix = ""
questions = [
    prompt_prefix + item.get('problem') for item in dataset
]

# Create dummy config
config = GRPOConfig(
    vllm_server_host="localhost",
    vllm_server_port=8000,
    repetition_penalty=1.0,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_completion_length=16384,
    vllm_guided_decoding_regex=None,
    eos_token="<|im_end|>",
    result_token="</tool_response>",
    assistant_token="<|im_start|>assistant",
)

random.shuffle(questions)

# Run tool-based inference and collect results
parallel = False
batch = True


if parallel:
    results = generate_with_tool_async(questions, config)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    dataset = load_dataset("json", data_files=output_path, split="train")
    dataset.push_to_hub("ryzax/qwen3_output_new")
    
elif batch:
    assert parallel == False
    results = []
    prompts = []
    for question in questions:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        question = tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(question)
        
    contexts, stats = generate_with_tool_batch(prompts, config)
    
    for (context, stat) in zip(contexts, stats):
        results.append({
            "system_prompt": SYSTEM_PROMPT,
            "question": question,
            "output": context,
            "tool_stats": stat
        })
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    dataset = load_dataset("json", data_files=output_path, split="train")
    dataset.push_to_hub("ryzax/qwen3_output_new")

else:
    results = []
    for i, question in enumerate(questions):
        # if i > 0:
        #     break
        print(f"\n=== Running Question {i + 1} ===")
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        question = tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # print(question)
        
        input_arg = {
            "question": question,
            "args": config
        }
        try:
            question, output, tool_stats = _generate_with_tool_safe(input_arg)
            results.append({
                "system_prompt": SYSTEM_PROMPT,
                "question": question,
                "output": output,
                "tool_stats": tool_stats
            })
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
                
                
            dataset = load_dataset("json", data_files=output_path, split="train")
            dataset.push_to_hub("ryzax/qwen3_output_new")

        except Exception as e:
            print(f"[ERROR] Failed on question {i + 1}: {str(e)}")
        


print(f"\n✅ Test complete. Output written to {output_path}")
