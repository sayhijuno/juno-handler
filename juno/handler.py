import os
import re
import sys
import time
import uuid
from types import SimpleNamespace

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams

from juno.schema import VALIDATIONS

# Model Settings
MODEL = os.getenv("MODEL_NAME")
DTYPE = os.getenv("MODEL_DTYPE")
QUANTIZATION = os.getenv("MODEL_QUANTIZATION")
TRUST_REMOTE_CODE = os.getenv("MODEL_TRUST_REMOTE_CODE", "").lower() in ("true", "1", "yes")
TOKENIZER = os.getenv("MODEL_TOKENIZER")
CONFIG_FORMAT = os.getenv("MODEL_CONFIG_FORMAT")
LOAD_FORMAT = os.getenv("MODEL_LOAD_FORMAT")

# Capacity Settings
MAX_MODEL_LEN = int(os.getenv("MODEL_MAX_LEN")) if os.getenv("MODEL_MAX_LEN") else None
MAX_NUM_SEQS = int(os.getenv("MODEL_MAX_NUM_SEQS")) if os.getenv("MODEL_MAX_NUM_SEQS") else None
DISTRIBUTED_EXECUTOR_BACKEND = os.getenv("DISTRIBUTED_EXECUTOR_BACKEND")

# Sampling defaults
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE") or "0.15")
DEFAULT_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS") or "32768")
DEFAULT_TOP_P = float(os.getenv("MODEL_TOP_P") or "0.95")

# Tool calling
TOOL_CALL_PARSER = os.getenv("MODEL_TOOL_CALL_PARSER") or None

model = None
tool_parser = None
_THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)

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

    think_match = _THINK_RE.search(text)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = (text[:think_match.start()] + text[think_match.end():]).strip()
    elif '<think>' in text:
        # Truncated — model hit max_tokens mid-thought
        idx = text.index('<think>')
        reasoning_content = text[idx + 7:].strip()
        text = text[:idx].strip()

    tool_calls = None
    if tool_parser is not None and job_input.get("tools"):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
        tools = [ChatCompletionToolsParam(**t) for t in job_input["tools"]]
        parsed = tool_parser.extract_tool_calls(
            text, SimpleNamespace(tools=tools)  # pyright: ignore[reportArgumentType]
        )
        if parsed.tools_called:
            tool_calls = [tc.model_dump() for tc in parsed.tool_calls]
            text = parsed.content or ""

    message = {
        "role": "assistant",
        "content": text,
    }

    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": os.getenv("RUNPOD_REQUEST_ID") or f"rp-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else output.finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids or []),
            "completion_tokens": len(output.token_ids or []),
            "total_tokens": len(result.prompt_token_ids or []) + len(output.token_ids or []),
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

    if TOOL_CALL_PARSER:
        from vllm.tool_parsers import ToolParserManager
        parser_cls = ToolParserManager.get_tool_parser(TOOL_CALL_PARSER)
        tool_parser = parser_cls(model.get_tokenizer())

    runpod.serverless.start({"handler": handler})
