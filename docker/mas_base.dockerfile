FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN apt-get update && apt-get install -y python3.11 python3-pip && pip install --upgrade pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN pip install poetry
RUN poetry install --no-root

CMD ["nvcc", "-V", "&&", "nvidia-smi"]
