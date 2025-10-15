# KCL

This repository provides the official evaluation implementation for Korean Canonical Legal Benchmark   

[![Datasets](https://img.shields.io/badge/🤗 Datasets-KCL-yellow?style=flat)](https://huggingface.co/datasets/lbox/kcl) [![Paper](https://img.shields.io/badge/arXiv-1234.1234-red?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/1234.1234)

## Usage

## Installation
```bash
git clone https://github.com/lbox-kr/kcl.git
cd kcl
uv sync
```

### Prepare Environment Settings Using .env
```
# for logging
HYDRA_FULL_ERROR=1

# for vertex
GOOGLE_APPLICATION_CREDENTIALS=.vertex_credentials.json

# for openai
OPENAI_API_KEY="sk-..."

# for bedrock
AWS_ACCESS_KEY_ID="..."
AWS_SECRET_ACCESS_KEY="..."
AWS_SESSION_TOKEN="..."
```

### Inference

## Inference

### KCL Essay

```bash
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_essay.yaml \
model_name=gemini-2.5-flash \
tasks_kwargs.with_precedents=True \
model_kwargs.thinking_budget=-1
```

### KCL MCQA

```bash
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_mcqa.yaml \
model_name=gemini-2.5-flash \
model_kwargs.thinking_budget=-1
```

### Evaluation

```bash
./scripts/eval/run_eval.sh \
./scripts/eval/configs/kcl_essay.yaml \
./outputs_infer/kcl_essay/gemini-2.5-flash/2025-10-15_10-04-43 \
n_jobs=8
```

### Citation
```tex

```
