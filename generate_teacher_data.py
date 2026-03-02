"""
Generate Teacher Responses for Response-Based Distillation
==========================================================
Uses gemma-3-27b-it to generate high-quality responses
for each prompt in the dataset, then saves as JSONL for SFT
fine-tuning of the student model.

Usage:
    python generate_teacher_data.py --config distill_config.yaml
    python generate_teacher_data.py --dataset_name tatsu-lab/alpaca --output_file teacher_data.jsonl

After generating, fine-tune student with:
    python sft_student.py --dataset teacher_data.jsonl
    (or use any SFT script / trl's SFTTrainer)
"""

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_generation_config(config_path: str) -> Dict[str, Any]:
    """Load config from YAML."""
    defaults = {
        "teacher_model": "google/gemma-3-27b-it",
        "dataset_name": "tatsu-lab/alpaca",
        "teacher_load_in_4bit": True,
        "teacher_bnb_4bit_compute_dtype": "bfloat16",
        "teacher_bnb_4bit_quant_type": "nf4",
        "generation": {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "batch_size": 4,
            "output_file": "./teacher_responses.jsonl",
        },
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f) or {}
        defaults.update(yaml_config)

    return defaults


def format_prompt(example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """Format a dataset example into a ChatML/Gemma prompt for the teacher."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")

    # For gemma, we want to enforce thinking in tags
    system_prompt = "You are a helpful assistant. You must first think about the answer and enclose your reasoning process within <reasoning> and </reasoning> tags. Then, provide your final answer within <answer> and </answer> tags."

    if input_text.strip():
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = f"{instruction}"

    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_content}"},
    ]
    
    # Use tokenizer to apply the correct Gemma chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    return prompt


def load_teacher_model(config: Dict[str, Any]):
    """Load the teacher model and tokenizer."""
    model_name = config["teacher_model"]
    logger.info(f"Loading teacher model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.get("teacher_load_in_4bit", True):
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
        compute_dtype = dtype_map.get(
            config.get("teacher_bnb_4bit_compute_dtype", "bfloat16"), torch.bfloat16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.get("teacher_bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model.eval()
    logger.info("Teacher model loaded successfully")
    return model, tokenizer


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    gen_config: Dict[str, Any],
) -> List[str]:
    """Generate responses for a batch of prompts."""
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_config.get("max_new_tokens", 1024),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    responses = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        generated = output[input_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(response)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate teacher responses for distillation")
    parser.add_argument("--config", type=str, default="distill_config.yaml")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--teacher_model", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    args = parser.parse_args()

    config = load_generation_config(args.config)

    # Apply CLI overrides
    if args.dataset_name:
        config["dataset_name"] = args.dataset_name
    if args.teacher_model:
        config["teacher_model"] = args.teacher_model

    gen_config = config.get("generation", {})
    if args.output_file:
        gen_config["output_file"] = args.output_file
    if args.batch_size:
        gen_config["batch_size"] = args.batch_size
    if args.max_new_tokens:
        gen_config["max_new_tokens"] = args.max_new_tokens

    output_file = gen_config.get("output_file", "./teacher_responses.jsonl")
    batch_size = gen_config.get("batch_size", 4)

    # --- Load dataset ---
    logger.info(f"Loading dataset: {config['dataset_name']}")
    if os.path.isfile(config["dataset_name"]):
        dataset = load_dataset("json", data_files=config["dataset_name"], split="train")
    else:
        dataset = load_dataset(config["dataset_name"], split="train")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    logger.info(f"Total examples: {len(dataset)}")

    # --- Load model ---
    model, tokenizer = load_teacher_model(config)

    # --- Generate ---
    logger.info(f"Generating responses (batch_size={batch_size})...")
    logger.info(f"Output file: {output_file}")

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    total = len(dataset)
    written = 0

    with open(output_file, "w", encoding="utf-8") as fout:
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch = dataset.select(range(start_idx, end_idx))

            # Format prompts
            prompts = []
            original_data = []
            for example in batch:
                prompt = format_prompt(example, tokenizer)
                prompts.append(prompt)
                original_data.append(dict(example))

            # Generate
            try:
                responses = generate_batch(model, tokenizer, prompts, gen_config)
            except Exception as e:
                logger.error(f"Error generating batch {start_idx}-{end_idx}: {e}")
                continue

            # Write results
            for orig, prompt, response in zip(original_data, prompts, responses):
                entry = {
                    "instruction": orig.get("instruction", ""),
                    "input": orig.get("input", ""),
                    "original_output": orig.get("output", ""),
                    "teacher_output": response,
                    "text": prompt + response,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1

            if (start_idx // batch_size + 1) % 10 == 0:
                logger.info(f"Progress: {end_idx}/{total} ({end_idx/total*100:.1f}%)")

    logger.info(f"✅ Done! Wrote {written} examples to {output_file}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Fine-tune student on teacher data:")
    logger.info(f"     python distill_gemma.py --dataset_name {output_file} --alpha 0.0")
    logger.info(f"     (alpha=0.0 means pure SFT on teacher responses)")
    logger.info(f"  2. Or use trl's SFTTrainer directly with the generated JSONL")


if __name__ == "__main__":
    main()
