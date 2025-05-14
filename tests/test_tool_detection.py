import json
from datasets import load_dataset
from trl.trainer.grpo_config import GRPOConfig
from trl.tools.async_tool_utils import generate_with_tool_safe, SYSTEM_PROMPT
from trl.tools.batch_tool_utils import batch_generate_with_tool
import random
from tqdm import tqdm
import multiprocessing


output_path = "/workspace/s1-tool/s1_tool/tools/jsonl/qwen3_output_5_13.json"


def _generate_with_tool_batch_parallel(questions: list, args: GRPOConfig, max_processes: int = 20) -> list:
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
        for question, output, tool_stats in pool.imap(generate_with_tool_safe, input_arguments):
            results.append({
                "system_prompt": SYSTEM_PROMPT,
                "question": question,
                "output": output,
                "tool_stats": tool_stats
            })
            bar.update(1)
            
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    dataset = load_dataset("json", data_files=output_path, split="train")
    dataset.push_to_hub("ryzax/qwen3_output_new")
            
    return results

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
    max_completion_length=128,
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
    _generate_with_tool_batch_parallel(questions, config)
    
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
        
    contexts, stats = batch_generate_with_tool(prompts, config)
    
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
            question, output, tool_stats = generate_with_tool_safe(input_arg)
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
