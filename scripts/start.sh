docker remove fibona-mas

docker run \
--env-file mas.env \
--name fibona-mas \
--volume /data/home/centonhuang/model/bge-m3:/app/model/bge-m3 \
--volume /data/home/centonhuang/model/bge-reranker-v2-m3:/app/model/bge-reranker-v2-m3 \
-p 9999:10000 \
mas:latest \
poetry run python -u main.py \
--embed /app/model/bge-m3 \
--embed_seq_len 4096 \
--rerank /app/model/bge-reranker-v2-m3 \
--rerank_seq_len 512