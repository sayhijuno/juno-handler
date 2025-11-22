# Juno Runpod Handler

A general-purpose vLLM handler for Runpod Serverless endpoints.

> **Warning**
>
> GGUF models are NOT supported.

## Usage

Send requests with your messages, optional tools, and sampling parameters:

```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024,
    "top_p": 0.9
  }
}
```

**On Runpod:**

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Say hello!"}]
    }
  }'
```

**Testing locally:**

```bash
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Say hello!"}]
    }
  }'
```

**OpenAI-compatible (via Runpod proxy):**

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_RUNPOD_API_KEY",
    base_url=f"https://api.runpod.ai/v2/{ENDPOINT_ID}/openai/v1"
)

response = client.chat.completions.create(
    model="unused",  # Model is set via MODEL_NAME environment variable
    messages=[{"role": "user", "content": "Say hello!"}]
)
```

## Configuration

**Required:**
- `MODEL_NAME` - HuggingFace model identifier

**Optional model settings:**
- `MODEL_QUANTIZATION` - awq, gptq, fp8 (valid vLLM arguments)
- `MODEL_MAX_LEN` - Maximum context length (reduces KV cache memory usage)
- `MODEL_TOKENIZER`, `MODEL_CONFIG_FORMAT`, `MODEL_LOAD_FORMAT` - only if your model needs them or vllm can't guess it

**Generation settings:**
- `MODEL_TEMPERATURE` (default: 0.15)
- `MAX_SAMPLING_TOKENS` (default: 32768)
- `TOP_P` (default: 0.95)

**Runtime:**
- `GPU_MEMORY_UTILIZATION` (default: 0.9)

## Models

Any HuggingFace model supported by vLLM works. For larger models, use quantized versions (look for `-AWQ` or `-GPTQ` suffix).

## Development

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for development instructions.

## License

MIT
