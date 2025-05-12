import json
from datasets import load_dataset
from trl.trainer.grpo_config import GRPOConfig
from trl.tools.qwen_tool_utils import generate_with_tool_safe, SYSTEM_PROMPT
import random

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
    repetition_penalty=1.5,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_completion_length=2048,
    vllm_guided_decoding_regex=None,
    eos_token="<|im_end|>",
    result_token="</tool_response>",
    assistant_token="<|im_start|>assistant",
)

random.shuffle(questions)

# Run tool-based inference and collect results
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
        output, tool_stats = generate_with_tool_safe(input_arg)
        results.append({
            "system_prompt": SYSTEM_PROMPT,
            "question": question,
            "output": output,
            "tool_stats": tool_stats
        })
        with open("/workspace/s1-tool/s1_tool/tools/jsonl/deepseek_output_5_11.json", "w") as f:
            json.dump(results, f, indent=4)
            
            
        dataset = load_dataset("json", data_files="/workspace/s1-tool/s1_tool/tools/jsonl/deepseek_output_5_11.json", split="train")
        dataset.push_to_hub("ryzax/deepseek_output_viewer")

    except Exception as e:
        print(f"[ERROR] Failed on question {i + 1}: {str(e)}")
        


print("\n✅ Test complete. Output written to qwen3_output_5_7.json")
