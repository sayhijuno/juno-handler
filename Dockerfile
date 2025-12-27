# Use specific version of nvidia cuda image
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# This is the location of the network volume for serverless endpoints.
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV VLLM_LOGGING_COLOR="0"

# NOTICE: This variable will cause uv sync to uninstall system dependencies that are not in the pyproject.toml file.
# This includes Jupyter, which is installed by default in the base image. 
# You can avoid this by unsetting UV_PROJECT_ENVIRONMENT to use a venv or switching to pip to manage system packages.
ENV UV_PROJECT_ENVIRONMENT=/usr/local

# Pick a folder, call it whatever you want - we call this project Juno.
WORKDIR /opt/juno

COPY pyproject.toml uv.lock ./

RUN uv sync --no-dev

COPY juno ./juno
CMD ["python", "-m", "juno.handler"]
