FROM ghcr.io/astral-sh/uv:debian
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update && apt-get install -y espeak-ng

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

CMD ["uv", "run", "main.py"]
