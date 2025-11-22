# AGENTS.md

This file provides guidance to AI agents and coding assistants working with this repository.

## What is This?

The Juno handler is a **Runpod serverless handler** for serving language models via vLLM.

### What is Runpod Serverless?

Runpod is a cloud provider that offers a serverless GPU platform. Unlike traditional serverless (AWS Lambda, etc.), Runpod is designed for GPU workloads:

- The user deploys a Docker container with a handler function (like this project)
- Runpod manages GPU workers that run your container
- Workers auto-scale based on request volume or the time requests spend in queue.
- Each request goes to an available worker; if none exist, one spins up (a traditional serverless cold start)
- Workers stay alive for a configurable time after requests to avoid cold starts

The key difference: you only pay when processing requests, not for idle time.

### What is a Handler?

A Runpod handler is powered by the runpod-python SDK and it is the entry point loaded by serverless workers that processes incoming requests. 

The pattern:

1. **Initialization phase** (once per worker): Load your model, download weights, prepare GPU memory
2. **Request phase** (per request): Receive JSON input, process it, return JSON output

The Juno handler loads a vLLM model once at startup, then processes chat completion requests using that loaded model. This amortizes the expensive model-loading across many requests for the lifecycle of this worker.

## Architecture

This is an intentionally minimal codebase, keeping it that way is crucial. We have two files:

- `juno/handler.py` - Handler implementation. Loads vLLM model at startup, exposes `handler()` function that Runpod calls
- `juno/schema.py` - Input validation schema (messages, tools, sampling parameters)

### Key Constraints

- **GGUF models NOT supported** - vLLM doesn't support GGUF format and this project does not offer any specific workarounds.

## Configuration Pattern

When adding settings:
- Prefix model-specific settings with `MODEL_` 
- Provide sane defaults or `None` to let vLLM use its defaults
- Never require configuration for optional features

## Additional Help

If you need specifics about Runpod Serverless APIs, endpoints, or deployment:
- Use the `SearchRunpodDocumentation` tool to search Runpod's documentation in real time. If you don't have access to this tool, ask your user to enable the MCP server at `https://docs.runpod.io/mcp`

## Development

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for practical development instructions, testing, and the AI contribution policy.
