FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY src /app/src
COPY main.py poetry.lock pyproject.toml /app/

RUN apt-get update && apt-get install -y python3.11 python3-pip && pip install --upgrade pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN pip install poetry
RUN poetry install --no-root
# RUN poetry install -E vllm -E reranker -E embedding -E awq -E gptq --no-root
