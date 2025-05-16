import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)

config = GRPOConfig(model_name_or_path="Qwen/Qwen3-4B")
config.tool_use = True

# Dummy completion_ids with tool response in the middle
#   prompt_ids will be ignored by loss already
#   completion_ids has: [a, b, <tool>, x, x, </tool>, c]
prompt_ids = torch.tensor([[2, 3]])  # 'a b'
completion_ids = torch.tensor([[2, 3, 100, 5, 5, 101, 4]])
completion_mask = torch.ones_like(completion_ids)

dataset = load_dataset("simplescaling/aime24_nofigures", split="train")

# Construct dummy trainer
# trainer = GRPOTrainer.__new__(GRPOTrainer)
trainer = GRPOTrainer(
    model=model_args.model_name_or_path,
    reward_funcs=reward_funcs,
    args=training_args,
    train_dataset=dataset[script_args.dataset_train_split],
    eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    callbacks=get_callbacks(training_args, model_args),
    processing_class=tokenizer,
)
trainer.args = config
trainer.tool_use = True
trainer.processing_class = tokenizer
trainer._tool_start_id = tokenizer.convert_tokens_to_ids("<tool_response>")
trainer._tool_end_id = tokenizer.convert_tokens_to_ids("</tool_response>")
trainer._build_tool_mask = GRPOTrainer._build_tool_mask.__get__(trainer)
trainer._get_per_token_logps = lambda *a, **kw: torch.ones_like(completion_ids, dtype=torch.float32)
trainer.beta = 0.0
trainer.temperature = 1.0
trainer.loss_type = "grpo"
trainer.accelerator = SimpleNamespace(
    gather_for_metrics=lambda x: x,
    unwrap_model=lambda x: x,
    is_main_process=True
)
trainer.model.training = True
trainer.ref_model = None
trainer._metrics = {"train": {}}

# Call compute_loss
inputs = {
    "prompt_ids": prompt_ids,
    "prompt_mask": torch.ones_like(prompt_ids),
    "completion_ids": completion_ids,
    "completion_mask": completion_mask * trainer._build_tool_mask(completion_ids),
    "advantages": torch.ones((1,), dtype=torch.float32),
    "old_per_token_logps": torch.zeros_like(completion_ids, dtype=torch.float32),
}
loss = trainer._compute_loss(trainer.model, inputs)

# Assert only non-tool tokens are used
masked_loss_positions = inputs["completion_mask"].squeeze().tolist()
expected_mask = [1, 1, 0, 0, 0, 0, 1]
assert masked_loss_positions == expected_mask, f"Mask mismatch: {masked_loss_positions}"
print("✅ Mask correctly applied:", masked_loss_positions)
print("✅ Loss computed without tool tokens:", loss.item())
