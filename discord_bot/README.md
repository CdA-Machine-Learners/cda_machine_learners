

# Running Discord Server

1. Copy config, fill out keys etc

2. Go!

```sh
python main.py
```

# Launching VLLM Server

```sh
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --model "$HOME/_/models/mistral/Mistral-7B-Instruct-v0.2/" --dtype
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --model "$HOME/_/bots/christian/mistral-bible-1/checkpoint-294-merged/" --dtype

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --model "$HOME/_/bots/christian/mistral-bible-1/checkpoint-294-merged/gptq" --load-format "safetensors" --quantization "gptq"



CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.api_server --model "$HOME/_/bots/christian/mistral-bible-2/checkpoint-594-merged" --dtype "bfloat16"

```
