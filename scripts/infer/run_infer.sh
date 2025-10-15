#!/bin/bash

source .venv/bin/activate

set -a
source .env
set +a

CONFIG_FILE=$1
shift

config_dir=$(dirname "$CONFIG_FILE")
config_name=$(basename "$CONFIG_FILE" .yaml)

python -m kcl.inference.infer \
  --config-dir "$config_dir" \
  --config-name "$config_name" \
  "$@"