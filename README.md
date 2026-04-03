# Voice OpenAI Server

FastAPI server providing OpenAI-compatible STT, TTS, and Chat endpoints.

## Features

- **STT**: Transcribe audio files using Cohere or Gemma4 models
- **TTS**: Generate speech from text using Kokoro models
- **Chat**: Text generation using Gemma4 (uses same model as STT)
- OpenAI-compatible API endpoints
- Supports multiple audio formats (WAV, FLAC, PCM)
- Automatic model loading and caching

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Run the server
uv run python main.py
```

The server starts on port 8000 by default. Set `PORT` environment variable to change:

```bash
PORT=3000 uv run python main.py
```

## API Endpoints

### List Models

```
GET /v1/models
```

Returns available STT, TTS, and LLM models.

### Transcribe Audio (STT)

```
POST /v1/audio/transcriptions
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file to transcribe |
| `model` | string | `""` | Model ID (uses first available if empty) |
| `language` | string | `"en"` | Language code |
| `response_format` | string | `"json"` | Response format: `json` or `text` |

Example using curl:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=CohereLabs/cohere-transcribe-03-2026"
```

### Generate Speech (TTS)

```
POST /v1/audio/speech
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text to synthesize |
| `model` | string | `""` | Model ID (uses first available if empty) |
| `voice` | string | `"af_heart"` | Voice to use |
| `response_format` | string | `"wav"` | Output format: `wav`, `flac`, or `pcm` |
| `speed` | float | `1.0` | Playback speed multiplier |

Example using curl:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","model":"hexgrad/Kokoro-82M"}' \
  --output speech.wav
```

### Chat Completions

```
POST /v1/chat/completions
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `""` | Model ID (uses first available if empty) |
| `messages` | array | required | Chat messages `[{role: "user", content: "..."}]` |
| `temperature` | float | `0.7` | Sampling temperature |
| `top_p` | float | `0.9` | Nucleus sampling threshold |
| `max_tokens` | int | `512` | Max tokens to generate |
| `stream` | bool | `false` | Enable streaming |

Example using curl:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"model":"google/gemma-4-E2B-it"}'
```

## Available Models

### STT Models

- **Gemma4**: `google/gemma-4-E2B-it` — Google's multimodal model (also supports chat)
- **Cohere**: `CohereLabs/cohere-transcribe-03-2026` — Cohere's speech recognition model

### Chat Models

- **Gemma4**: `google/gemma-4-E2B-it` — Uses same model instance as STT

### TTS Models

- **Kokoro**: `hexgrad/Kokoro-82M` — High-quality text-to-speech model

## Development

```bash
# Install dev dependencies
uv sync

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Run tests (when configured)
uv run pytest
```

See [AGENTS.md](AGENTS.md) for detailed code style guidelines and conventions.
