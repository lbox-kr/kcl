# KCL

This repository provides the official evaluation implementation for Korean Canonical Legal Benchmark   

[![Datasets](https://img.shields.io/badge/🤗 Datasets-KCL-yellow?style=flat)](https://huggingface.co/datasets/lbox/kcl) [![Paper](https://img.shields.io/badge/arXiv-1234.1234-red?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/1234.1234)

### Installation
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

## KCL Essay

### Inference

```bash
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_essay.yaml \
model_name=gemini-2.5-flash \
tasks_kwargs.with_precedents=True \
model_kwargs.thinking_budget=-1
```

### Evaluation

```bash
./scripts/eval/run_eval.sh \
./scripts/eval/configs/kcl_essay.yaml \
./outputs_infer/kcl_essay/gemini-2.5-flash/2025-10-15_10-04-43 \
n_jobs=8
```

### Results

<img width="339" height="597" alt="fEb_RSiHVCGT6v0V7A13B" src="https://github.com/user-attachments/assets/c689fdb8-b89e-4c86-8232-0a871f40c448" />


## KCL MCQA

### Inference

```bash
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_mcqa.yaml \
model_name=gemini-2.5-flash \
model_kwargs.thinking_budget=-1
```

### Evaluation

```bash
./scripts/eval/run_eval.sh \
./scripts/eval/configs/kcl_mcqa.yaml \
./outputs_infer/kcl_mcqa/gemini-2.5-flash/2025-10-15_12-33-09 \
n_jobs=8
```

### Results

<img width="339" height="616" alt="OmiTG5Tv6pN2PRtiBhspy" src="https://github.com/user-attachments/assets/f6326505-5611-4c66-a80d-a458549b3730" />

## Citation

```tex
@misc{
    kcl,
    title={Korean Canonical Legal Benchmark: Toward Knowledge-Independent Evaluation of LLMs' Legal Reasoning Capabilities}, 
    author={Hongseok Oh, Wonseok Hwang, and Kyoung-Woon On},
    year={2025}
}
```

## License

Our evaluation code and dataset are licensed under the [CC BY-NC 4.0 license](LICENSE.md).
