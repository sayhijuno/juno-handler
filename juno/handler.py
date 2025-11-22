import os

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams

from juno.schema import VALIDATIONS

MODEL = os.getenv("MODEL_NAME")
TOKENIZER = os.getenv("MODEL_TOKENIZER")
CONFIG_FORMAT = os.getenv("MODEL_CONFIG_FORMAT")
LOAD_FORMAT = os.getenv("MODEL_LOAD_FORMAT")
QUANTIZATION = os.getenv("MODEL_QUANTIZATION")
MAX_MODEL_LEN = os.getenv("MODEL_MAX_LEN")

DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.15"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_SAMPLING_TOKENS", "32768"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.95"))

if not MODEL:
    print("Define a MODEL_NAME...")
    os._exit(-1)

log.info("Loading {}...".format(MODEL))

model = LLM(
    model=MODEL,
    tokenizer_mode=TOKENIZER if TOKENIZER else "auto",
    config_format=CONFIG_FORMAT if CONFIG_FORMAT else "auto",
    load_format=LOAD_FORMAT if LOAD_FORMAT else "auto",
    quantization=QUANTIZATION if QUANTIZATION else None,
    max_model_len=int(MAX_MODEL_LEN) if MAX_MODEL_LEN else None,
    tensor_parallel_size=int(os.getenv("RUNPOD_GPU_COUNT", "1")),
    gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
)


def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)

    if "errors" in input_validation:
        return {"error": input_validation["errors"]}
    job_input = input_validation["validated_input"]

    temperature = job_input.get("temperature")
    max_tokens = job_input.get("max_tokens")
    top_p = job_input.get("top_p")
    
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

    generated_text = ""

    for chunk in model_output:
        for output in chunk.outputs:
            generated_text += output.text

    return generated_text


runpod.serverless.start({"handler": handler})
