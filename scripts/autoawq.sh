python ./scripts/autoawq.py \
--model_path Qwen/Qwen1.5-4B-Chat \
--quant_path Qwen/Qwen1.5-4B-gptq-4bit-Chat \
--dataset_path alpaca_gpt4_data_zh.json \
--text_column output \
--num_samples 10000