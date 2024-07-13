# Aris-AI-Model-Server

[ [English](README.md) | 简体中文 ]

## 介绍

在AI应用开发中，我们往往需要部署多种模型来完成不同的任务，对于模型对话服务，我们需要用到LLM模型，对于知识库检索服务，我们需要用到Embedding、Reranker模型。因此`Aris-AI-Model-Server`应运而生，它侧重于集成多种模型服务于一体，为使用者提供简单、方便的模型接入能力。项目名来源于Blue Archive中的角色Aris，如下图

---

<p align="center">
  <img src="assets/93331597.png" width="50%">
  <br>Aris: Blue Archive 中的角色
</p>

---

## 更新日志

- [2024-07-13] Aris-AI-Model-Server正式开源。

- [2024-06-23] 我们发布了[Aris-14B-Chat系列模型](https://huggingface.co/collections/Aris-AI/aris-chat-arcturus-6642fd11069310a4467db222)，该模型基于[Qwen1.5-14B-Chat](https://huggingface.co/Qwen/Qwen1.5-14B-Chat)在我们的140K条私有数据集进行了SFT和DPO。在使用该模型时,请遵守Qwen开源协议。

## 技术栈

### 模型后端

#### Embedding

- Sentence Transformers

#### Reranker

- Sentence Transformers

#### LLM

- VLLM
- MLX

### API后端

- FastAPI

## API接口

| 路由 | 请求方法 | 鉴权 | OpenAI Compatible| 描述 |
| --- | --- | --- | --- | --- |
| / | GET | ❌ | ❌ | 根目录 |
| /v1/embeddings | GET | ✅ | ❌ | 获取所有Embedding模型 |
| /v1/embeddings | POST | ✅ | ✅ | 调用Embedding进行文本嵌入 |
| /v1/rerankers | GET | ✅ | ❌ | 获取所有Reranker模型 |
| /v1/rerankers | POST | ✅ | ❌ | 调用Reranker进行文档重排 |
| /v1/models | GET | ✅ | ✅ | 获取所有LLM |
| /v1/chat/completions | POST | ✅ | ✅ | 调用LLM进行对话生成 |

## 项目结构

```text
.
├── assets
│   └── 110531412.jpg
├── config # 环境变量和模型配置
│   ├── .env.template
│   └── models.yaml.template
├── dockerfile
├── main.py
├── poetry.lock
├── pyproject.toml
├── scripts # awq、gptq量化脚本
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
    │   ├── arg.py # 命令行参数
    │   ├── env.py # 环境变量
    │   ├── gbl.py # 全局变量
    │   ├── __init__.py
    │   └── model.py # 模型配置
    ├── controller
    │   ├── controller.py # Engine控制器
    │   └── __init__.py
    ├── engine # 模型调用引擎
    │   ├── base.py
    │   ├── embedding.py
    │   ├── mlx.py
    │   ├── reranker.py
    │   └── vllm.py
    ├── logger # 日志库
    │   └── __init__.py
    ├── middleware # 中间件
    │   └── logger
    │       └── __init__.py
    └── utils
        ├── formatter.py # prompt格式(参考自llama-factory的实现)
        └── template.py # 格式(参考自llama-factory的实现)
```

## 本地部署

### 克隆仓库

```bash
git clone https://github.com/hcd233/Aris-AI-Model-Server.git
cd Aris-AI-Model-Server
```

### 创建虚拟环境（可选）

可以不创建，但是需要确保python环境为3.11

```bash
conda create -n aris python=3.11.0
conda activate aris
```

### 安装依赖

#### 安装poetry

```bash
pip install poetry
```

#### 根据需求安装依赖

| 依赖 | 描述 | 命令 |
| --- | --- | --- |
| base | 安装API启动的基础依赖 | `poetry install` |
| reranker | 安装部署reranker模型的依赖 | `{{base}}` + `-E reranker` |
| embedding | 安装部署embedding模型的依赖 | `{{base}}` + `-E embedding` |
| vllm | 安装vllm后端的依赖 | `{{base}}` + `-E vllm` |
| mlx | 安装mlx后端的依赖 | `{{base}}` + `-E mlx` |
| awq | 安装awq量化的依赖 | `{{base}}` + `-E awq` |
| gptq | 安装gptq量化的依赖 | `{{base}}` + `-E gptq` |

举例：如果我希望部署embedding模型，还有用awq量化模型后用vllm部署模型，那么我需要执行以下命令安装依赖：

```bash
poetry install -E embedding -E awq -E vllm
```

### 配置model.yaml和.env（略）

具体修改内容请参考template文件

```bash
cp config/models.yaml.template models.yaml
cp config/.env.template .env
```

### 启动API

```bash
python main.py --config_path models.yaml
```

### 模型量化

#### awq

```bash
bash scripts/autoawq.sh
```

#### awq

```bash
bash scripts/autogptq.sh
```


## Docker部署

暂无

## 项目展望

### 目标

1. 架构划分：由单机版本拓展为基于kubernetes的分布式版本
2. 丰富后端：支持更多模型后端，如Triton、ONNX等

### 作者状态

因为工作繁忙，项目进度可能会比较慢，随缘更新一下，欢迎PR和Issue
