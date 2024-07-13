import json
import random
from pathlib import Path

from awq import AutoAWQForCausalLM
from fire import Fire
from loguru import logger
from transformers import AutoTokenizer


def quantize_autoawq(model_path: str, quant_path: str, dataset_path: str, text_column: str, num_samples: int = 1000) -> None:
    if model_path == quant_path:
        logger.error("[Check Path] model_path and quant_path should not be the same")
        exit(-1)

    # Load Dataset
    _dataset_path = Path(dataset_path)
    if not _dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if _dataset_path.suffix == ".json":
        dataset = json.loads(_dataset_path.read_text(encoding="utf-8"))
    elif _dataset_path.suffix == ".jsonl":
        dataset = [json.loads(line) for line in _dataset_path.read_text(encoding="utf-8").splitlines()]
    else:
        raise ValueError(f"Dataset file format not supported: {dataset_path}")

    dataset = random.sample([data[text_column] for data in dataset], min(num_samples, len(dataset)))
    logger.info(f"[Load Dataset] load {len(dataset)} samples from {dataset_path}")

    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    logger.info(f"[Quantize AutoAWQ] quantization config: {quant_config}")

    logger.info(f"[Load Model] loading model from {model_path}")
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.success(f"[Load Model] load model from {model_path} successfully")

    # Quantize
    logger.info("[Quantize Model] quantizing model...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=dataset, text_column=text_column)
    logger.success("[Quantize Model] quantization done")

    # Save quantized model
    logger.info(f"[Save Model] save quantized model to {quant_path}")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    logger.success(f"[Save Model] save quantized model to {quant_path} successfully")


if __name__ == "__main__":
    Fire(quantize_autoawq)
