# Aris-AI-Model-Server

[ English | [简体中文](README_zh.md) ]

## Introduction

In AI application development, we often need to deploy multiple models to complete different tasks. For model dialogue services, we need LLM models, and for knowledge base retrieval services, we need Embedding and Reranker models. Therefore, `Aris-AI-Model-Server` was born, focusing on integrating multiple model services into one, providing users with simple and convenient model access capabilities. The project name comes from the character Aris in Blue Archive, as shown in the figure below:

---

<p align="center">
  <img src="assets/93331597.png" width="50%">
  <br>Aris: Character from Blue Archive
</p>

---

## Changelog

- [2024-07-13] Aris-AI-Model-Server officially open-sourced.

- [2024-06-23] We released the [Aris-14B-Chat series models](https://huggingface.co/collections/Aris-AI/aris-chat-arcturus-6642fd11069310a4467db222), which are based on [Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat) and have undergone SFT and DPO on our private dataset of 140K entries. When using this model, please comply with the Qwen open-source license.

## Technology Stack

### Model Backend

#### Embedding

- Sentence Transformers

#### Reranker

- Sentence Transformers

#### LLM

- VLLM
- MLX

### API Backend

- FastAPI

## API Interfaces

| Route | Request Method | Authentication | OpenAI Compatible | Description |
| --- | --- | --- | --- | --- |
| / | GET | ❌ | ❌ | Root directory |
| /v1/embeddings | GET | ✅ | ❌ | Get all Embedding models |
| /v1/embeddings | POST | ✅ | ✅ | Call Embedding for text embedding |
| /v1/rerankers | GET | ✅ | ❌ | Get all Reranker models |
| /v1/rerankers | POST | ✅ | ❌ | Call Reranker for document reranking |
| /v1/models | GET | ✅ | ✅ | Get all LLMs |
| /v1/chat/completions | POST | ✅ | ✅ | Call LLM for dialogue generation |

## Project Structure

```text
.
├── assets
│   └── 110531412.jpg
├── config # Environment variables and model configuration
│   ├── .env.template
│   └── models.yaml.template
├── dockerfile
├── main.py
├── poetry.lock
├── pyproject.toml
├── scripts # awq, gptq quantization scripts
│   ├── autoawq.py
│   ├── autoawq.sh
│   ├── autogptq.py
│   └── autogptq.sh
└── src
    ├── api # OpenAI Compatible API
    │   ├── auth
    │   │   └── bearer.py
    │   ├── model
    │   │   ├── chat_cmpl.py
    │   │   ├── embedding.py
    │   │   ├── reranker.py
    │   │   └── root.py
    │   └── router
    │       ├── __init__.py
    │       ├── root.py
    │       └── v1
    │           ├── chat_cmpl.py
    │           ├── embedding.py
    │           ├── __init__.py
    │           └── reranker.py
    ├── config
    │   ├── arg.py # Command line arguments
    │   ├── env.py # Environment variables
    │   ├── gbl.py # Global variables
    │   ├── __init__.py
    │   └── model.py # Model configuration
    ├── controller
    │   ├── controller.py # Engine controller
    │   └── __init__.py
    ├── engine # Model invocation engine
    │   ├── base.py
    │   ├── embedding.py
    │   ├── mlx.py
    │   ├── reranker.py
    │   └── vllm.py
    ├── logger # Logging library
    │   └── __init__.py
    ├── middleware # Middleware
    │   └── logger
    │       └── __init__.py
    └── utils
        ├── formatter.py # Prompt format (referenced from llama-factory implementation)
        └── template.py # Format (referenced from llama-factory implementation)
```

## Local Deployment

### Clone Repository

```bash
git clone https://github.com/hcd233/Aris-AI-Model-Server.git
cd Aris-AI-Model-Server
```

### Create Virtual Environment (Optional)

This step is optional, but ensure your Python environment is 3.11

```bash
conda create -n aris python=3.11.0
conda activate aris
```

### Install Dependencies

#### Install poetry

```bash
pip install poetry
```

#### Install Dependencies Based on Requirements

| Dependency | Description | Command |
| --- | --- | --- |
| base | Install basic dependencies for API startup | `poetry install` |
| reranker | Install dependencies for deploying reranker models | `{{base}}` + `-E reranker` |
| embedding | Install dependencies for deploying embedding models | `{{base}}` + `-E embedding` |
| vllm | Install dependencies for vllm backend | `{{base}}` + `-E vllm` |
| mlx | Install dependencies for mlx backend | `{{base}}` + `-E mlx` |
| awq | Install dependencies for awq quantization | `{{base}}` + `-E awq` |
| gptq | Install dependencies for gptq quantization | `{{base}}` + `-E gptq` |

Example: If you want to deploy an embedding model, use awq quantization, and deploy models with vllm, execute the following command to install dependencies:

```bash
poetry install -E embedding -E awq -E vllm
```

### Configure model.yaml and .env (Omitted)

Please refer to the template files for specific modifications

```bash
cp config/models.yaml.template models.yaml
cp config/.env.template .env
```

### Start API

```bash
python main.py --config_path models.yaml
```

### Model Quantization

#### awq

```bash
bash scripts/autoawq.sh
```

#### gptq

```bash
bash scripts/autogptq.sh
```

## Docker Deployment

Not available yet

## Project Outlook

### Goals

1. Architecture division: Expand from single-machine version to kubernetes-based distributed version
2. Enrich backends: Support more model backends, such as Triton, ONNX, etc.

### Author Status

Due to busy work, project progress may be slow, updates will be occasional. PRs and Issues are welcome.
