# Korean Canonical Legal Benchmark (KCL)

This repository provides the official evaluation implementation for Korean Canonical Legal Benchmark   

[![Datasets](https://img.shields.io/badge/🤗 Datasets-KCL-yellow?style=flat)](https://huggingface.co/datasets/lbox/kcl) [![Paper](https://img.shields.io/badge/arXiv-2512.24572-red?style=flat&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2512.24572)

🎉 Our paper has been accepted to the 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2026, main, short.)!

## Why KCL?

KCL is designed to **disentangle knowledge coverage from evidence-grounded reasoning**.   

KCL supports two complementary evaluation axes:
1. **Knowledge Coverage**: performance without extra context.  
2. **Evidence-Grounded Reasoning**: performance **with per-question supporting precedents** provided in-context.

For essay questions, KCL further offers **instance-level rubrics** to enable **LLM-as-a-Judge** automated scoring.

For more information, please refer to our paper 

### Intended Uses
 - Separating knowledge vs. reasoning by comparing vanilla and with-precedent settings.
 - Legal RAG research using question-aligned gold precedents to establish retriever/reader upper bounds.
 - Fine-grained feedback via rubric-level diagnostics on essay outputs.

## Components

- **KCL-Essay** (open-ended generation)  
  - 169 questions, 550 supporting precedents, 2,739 instance-level rubrics.
- **KCL-MCQA** (five-choice question answering)  
  - 283 questions, 1,103 supporting precedents.
 
## Usage

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
tasks_kwargs.with_precedents=True
# model_kwargs.thinking_budget=-1  # Optional: OAIModel defaults to "medium"
```

### Evaluation

```bash
./scripts/eval/run_eval.sh \
./scripts/eval/configs/kcl_essay.yaml \
./outputs_infer/kcl_essay/gemini-2.5-flash/2025-10-15_10-04-43 \
n_jobs=8
```

#### Cost

Evaluation Cost Statistics (KCL-Essay)
 * 169 questions – Gemini Flash token usage
 * Input: 6,896,400 tokens × $0.30 / 1M
 * Output: 780,173 tokens × $2.50 / 1M
 * Approximate cost per run: less than $5 -> Using caching, the cost can be reduced further.

## KCL MCQA

### Inference

```bash
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_mcqa.yaml \
model_name=gemini-2.5-flash
# model_kwargs.thinking_budget=-1  # Optional: OAIModel defaults to "medium"
```

### Evaluation

```bash
./scripts/eval/run_eval.sh \
./scripts/eval/configs/kcl_mcqa.yaml \
./outputs_infer/kcl_mcqa/gemini-2.5-flash/2025-10-15_12-33-09 \
n_jobs=8
```

## For Local Model

The evaluation code assumes a locally hosted internal model exposed via an OpenAI-compatible API.   
The local model is configured using a YAML file, as shown below:   
```yaml
model_name: "google/gemma-3-27b-it"
model_base_name: "gemma-3-27b-it"
model_kwargs:
  port: 8000

tasks: kcl_{essay|mcqa}
tasks_kwargs:
  with_precedents: False

n_jobs: 8
verbose: False

hydra:
  run:
    dir: outputs_infer/${tasks}/${model_base_name}/${now:%Y-%m-%d_%H-%M-%S}
```
Save this configuration file as:   
`scripts/infer/configs/kcl_{mcqa|essay}_local.yaml`

Then, run the inference using the same command as follows:
```
./scripts/infer/run_infer.sh \
./scripts/infer/configs/kcl_{mcqa|essay}_local.yaml
```

**Note:** The evaluation script allows model directory names with suffixes (e.g., `model_name_no_reasoning`). The directory name only needs to start with the base model name.

## Citation

```tex
@inproceedings{
    oh2026korean,
    title={Korean Canonical Legal Benchmark: Toward Knowledge-Independent Evaluation of {LLM}s' Legal Reasoning Capabilities},
    author={Hongseok Oh and Wonseok Hwang and Kyoung-Woon On},
    booktitle={19th Conference of the European Chapter of the Association for Computational Linguistics},
    year={2026},
    url={https://openreview.net/forum?id=Dw0sFP4l5s}
}
```

## License

Our evaluation code and dataset are licensed under the [CC BY-NC 4.0 license](LICENSE.md).
