import json
import random

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import Dataset
from fire import Fire
from loguru import logger
from transformers import AutoTokenizer


def load_data(data_path, tokenizer, num_samples, ):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(num_samples, len(raw_data)))

    def _dummy_gen():
        return raw_data

    def _tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(_dummy_gen)

    dataset = dataset.map(
        _tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return dataset


def quantize_autogptq(model_path: str, quant_path: str, alpaca_dataset: str, batch_size: int = 16, num_samples: int = 10000) -> None:
    if model_path == quant_path:
        logger.error("[Check Path] model_path and quant_path should not be the same")
        exit(-1)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = load_data(alpaca_dataset, tokenizer, num_samples)

    quantize_config = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True)
    logger.info(f"[Quantize AutoGPTQ] quantization config: {quantize_config}")

    logger.info(f"[Load Model] loading model from {model_path}")
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=quantize_config, trust_remote_code=True)
    logger.success(f"[Load Model] load model from {model_path} successfully")
    
    # Quantize
    logger.info("[Quantize Model] quantizing model...")
    model.quantize(examples=dataset, batch_size=batch_size)
    logger.success("[Quantize Model] quantization done")

    # Save quantized model
    logger.info(f"[Save Model] save quantized model to {quant_path}")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    logger.success(f"[Save Model] save quantized model to {quant_path} successfully")


if __name__ == "__main__":
    Fire(quantize_autogptq)
