#!/bin/bash

source .venv/bin/activate

set -a
source .env
set +a

CONFIG_FILE=$1
INPUT_DIR=$2

shift 2

config_dir=$(dirname "$CONFIG_FILE")
config_name=$(basename "$CONFIG_FILE" .yaml)

infer_timestamp=$(basename "$INPUT_DIR")
model_name=$(basename "$(dirname "$INPUT_DIR")")
task_name=$(basename "$(dirname "$(dirname "$INPUT_DIR")")")

hydra_run_dir="outputs_eval/$task_name/$model_name/$infer_timestamp/\${now:%Y-%m-%d_%H-%M-%S}"

python -m kcl.evaluation.eval \
  --config-dir "$config_dir" \
  --config-name "$config_name" \
  input_dir="$INPUT_DIR" \
  hydra.run.dir="$hydra_run_dir" \
  "$@"