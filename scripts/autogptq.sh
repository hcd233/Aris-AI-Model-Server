python ./scripts/autogptq.py \
--model_path Qwen/Qwen1.5-4B-Chat \
--quant_path Qwen/model/Qwen1.5-4B-gptq-4bit-Chat \
--alpaca_dataset alpaca_gpt4_data_zh.json \
--batch_size 4 \
--num_samples 5000