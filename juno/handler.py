import os
import re
import sys
import time
import uuid

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams

from juno.schema import VALIDATIONS

MODEL = os.getenv("MODEL_NAME")
DTYPE = os.getenv("MODEL_DTYPE")
QUANTIZATION = os.getenv("MODEL_QUANTIZATION")
TRUST_REMOTE_CODE = os.getenv("MODEL_TRUST_REMOTE_CODE", "").lower() in ("true", "1", "yes")
TOKENIZER = os.getenv("MODEL_TOKENIZER")
CONFIG_FORMAT = os.getenv("MODEL_CONFIG_FORMAT")
LOAD_FORMAT = os.getenv("MODEL_LOAD_FORMAT")

MAX_MODEL_LEN = int(os.getenv("MODEL_MAX_LEN")) if os.getenv("MODEL_MAX_LEN") else None
MAX_NUM_SEQS = int(os.getenv("MODEL_MAX_NUM_SEQS")) if os.getenv("MODEL_MAX_NUM_SEQS") else None
DISTRIBUTED_EXECUTOR_BACKEND = os.getenv("DISTRIBUTED_EXECUTOR_BACKEND")

DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE") or "0.15")
DEFAULT_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS") or "32768")
DEFAULT_TOP_P = float(os.getenv("MODEL_TOP_P") or "0.95")

model = None

def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)

    if "errors" in input_validation:
        return {"error": {"type": "validation_error", "message": "Invalid input", "details": input_validation["errors"]}}
    job_input = input_validation["validated_input"]

    messages = job_input.get("messages")
    prompt = job_input.get("prompt")
    temperature = job_input.get("temperature")
    max_tokens = job_input.get("max_tokens")
    top_p = job_input.get("top_p")

    if messages and prompt:
        return {"error": {"type": "validation_error", "message": "Provide either 'messages' or 'prompt', not both"}}

    if not messages and not prompt:
        return {"error": {"type": "validation_error", "message": "Either 'messages' or 'prompt' is required"}}

    if prompt:
        job_input["messages"] = [{"role": "user", "content": prompt}]
    
    sampler = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        top_p=top_p if top_p is not None else DEFAULT_TOP_P,
    )

    model_output = model.chat(
        messages=job_input["messages"],
        sampling_params=sampler,
        use_tqdm=False,
        chat_template_content_format="string",
        tools=job_input.get("tools", None),
    )

    result = model_output[0]
    output = result.outputs[0]

    text = output.text
    reasoning_content = None

    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    message = {
        "role": "assistant",
        "reasoning_content": reasoning_content,
        "content": text,
    }

    if hasattr(output, 'tool_calls') and output.tool_calls:
        message["tool_calls"] = output.tool_calls

    return {
        "id": os.getenv("RUNPOD_REQUEST_ID") or f"rp-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": output.finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(output.token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(output.token_ids),
        }
    }


if __name__ == '__main__':
    if not MODEL:
        print("Define a MODEL_NAME...")
        sys.exit(1)

    log.info("Loading {}...".format(MODEL))

    model = LLM(
        model=MODEL,
        tokenizer_mode=TOKENIZER or "auto",
        config_format=CONFIG_FORMAT or "auto",
        load_format=LOAD_FORMAT or "auto",
        quantization=QUANTIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE or "auto",
        trust_remote_code=TRUST_REMOTE_CODE,
        max_num_seqs=MAX_NUM_SEQS,
        distributed_executor_backend=DISTRIBUTED_EXECUTOR_BACKEND,
        tensor_parallel_size=int(os.getenv("RUNPOD_GPU_COUNT") or "1"),
        gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION") or "0.8"),
    )

    runpod.serverless.start({"handler": handler})
    runpod.serverless.start({"handler": handler})
