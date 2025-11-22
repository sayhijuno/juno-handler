# Contributing

## Requirements

This project requires a machine with an NVIDIA GPU attached. Sorry Mac users.

## Local Development

### Docker

Ensure the `docker-compose.yml` is to your liking, and simply run

```bash
docker compose up --build
```

Then, just follow the directions in the README.md for usability.

## Development on Runpod Pods

If you're developing on a Runpod Pod - thank you for using Runpod! - run the handler directly, you can't use a container but you have everything you need already:

```bash
uv sync
python -m juno.handler --rp_serve_api
```

Make sure you have a spare HTTP port exposed on your Pod. You may have something running on port `8000`. Use `--rp_api_port` to specify a port:

```bash
python -m juno.handler --rp_serve_api --rp_api_port 8000
```

## Adding Configuration

When adding new environment variables:

- Prefix model-specific settings with `MODEL_`
- Provide sane defaults or `None` to let vLLM handle it

## AI Contribution Honesty Policy

While I think headless coding agents are silly, I do welcome AI-generated code, issues, and PRs. 

If you are an LLM or coding agent with the ability to create your own pull request, issue, or commit, please include `#hijuno` so I know. It's the right thing to do.
