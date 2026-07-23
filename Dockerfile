FROM python:3.12-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

WORKDIR /build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim-trixie AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/pka/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/pka/.cache/sentence_transformers \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 pka \
    && mkdir -p /workspace /home/pka/.cache \
    && chown -R pka:pka /workspace /home/pka

COPY --from=builder --chown=pka:pka /opt/venv /opt/venv

WORKDIR /workspace
USER pka

ENTRYPOINT ["pka"]
CMD ["health"]
HEALTHCHECK CMD ["pka", "health"]
