# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Prefix-sliding SFT script. Trains a Qwen3 model on OpenThoughts-114k (metadata config) with prefix sliding
# enabled: every token attends to the prompt (prefix) plus a sliding window of `prefix_sliding_window` recent tokens.
#
# Example (single GPU):
#   accelerate launch examples/scripts/sft_prefix_sliding.py \
#       --config examples/cli_configs/sft_prefix_sliding_qwen3_1.7b.yaml
#
# The dataset is mapped in-memory from the `metadata` config (`problem` -> prompt, `deepseek_reasoning` +
# `deepseek_solution` -> completion) into the `{prompt, completion}` format expected by `SFTTrainer`.

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)


# --- OpenThoughts-114k -> prompt/completion formatting (edit as needed) --------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Reason step by step inside your thinking, "
    "then provide a concise final answer."
)
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def build_prompt_completion(example, tokenizer):
    """Map one OpenThoughts-114k (metadata) example to `{prompt, completion}`.

    `prompt` is the task/instruction formatted with the model's chat template and an open assistant turn; this is the
    "prefix" that prefix sliding keeps fully attended. `completion` is the DeepSeek reasoning trace plus final answer.
    """
    problem = example["problem"]
    reasoning = (example.get("deepseek_reasoning") or "").strip()
    solution = (example.get("deepseek_solution") or example.get("ground_truth_solution") or "").strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Qwen3 supports `enable_thinking`; other tokenizers may not.
        chat_kwargs["enable_thinking"] = True
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)
    except (TypeError, ValueError):
        chat_kwargs.pop("enable_thinking", None)
        prompt = tokenizer.apply_chat_template(messages, **chat_kwargs)

    completion = f"{THINK_OPEN}\n{reasoning}\n{THINK_CLOSE}\n\n{solution}"
    return {"prompt": prompt, "completion": completion}


# --- Entrypoint -------------------------------------------------------------------------------------


def main(script_args, training_args, model_args):
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def format_example(example):
        return build_prompt_completion(example, tokenizer)

    train_dataset = dataset[script_args.dataset_train_split].map(
        format_example, remove_columns=dataset[script_args.dataset_train_split].column_names
    )
    eval_dataset = None
    if training_args.eval_strategy != "no":
        eval_dataset = dataset[script_args.dataset_test_split].map(
            format_example, remove_columns=dataset[script_args.dataset_test_split].column_names
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft-prefix-sliding", help="Run prefix-sliding SFT", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
