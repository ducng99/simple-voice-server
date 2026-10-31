FROM ghcr.io/astral-sh/uv:alpine
ENV UV_LINK_MODE=copy

WORKDIR /app

RUN apk add --no-cache espeak-ng

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

CMD ["uv", "run", "main.py"]
