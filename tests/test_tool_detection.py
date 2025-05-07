import json
from datasets import load_dataset
from ..trl.trainer.grpo_config import GRPOConfig
from ..trl.tools.qwen_tool_utils import _generate_with_tool_safe

# Load a small number of test samples
dataset = load_dataset("simplescaling/aime24_nofigures", split="train[:3]")  # only 3 for quick test

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
    temperature=0.7,
    top_p=0.95,
    top_k=None,
    min_p=0.0,
    max_completion_length=4096,
    vllm_guided_decoding_regex=None,
    eos_token="<|endoftext|>",
    result_token="</tool_response>",
    python_token="<tool_call>{\"name\": \"python\"",
    retrieval_token="<tool_call>{\"name\": \"retrieval\"",
    saving_token="<tool_call>{\"name\": \"saving\"",
    assistant_token="<|im_start|>assistant",
    thinking_tokens=["<think>"]
)

# Run tool-based inference and collect results
results = []

for i, question in enumerate(questions):
    print(f"\n=== Running Question {i + 1} ===")
    input_arg = {
        "question": question,
        "args": config
    }
    try:
        output, tool_stats = _generate_with_tool_safe(input_arg)
        results.append({
            "question": question,
            "output": output,
            "tool_stats": tool_stats
        })
    except Exception as e:
        print(f"[ERROR] Failed on question {i + 1}: {str(e)}")

# Save results for inspection
with open("inftythink_tool_test.json", "w") as f:
    json.dump(results, f, indent=4)

print("\n✅ Test complete. Output written to inftythink_tool_test.json.")
