# AGENTS.md — Voice OpenAI Server

## Project Overview

FastAPI server providing OpenAI-compatible STT and TTS endpoints. Uses `uv` for dependency management, Python 3.12.

## Commands

```bash
# Install dependencies
uv sync

# Run the server
python main.py
# or
uv run python main.py

# Run with specific host/port (uses PORT env var, default 8000)
PORT=3000 uv run python main.py
```

### Testing

No test framework is currently configured. To add tests:

```bash
# Install pytest
uv add --dev pytest httpx

# Run all tests
uv run pytest

# Run a single test
uv run pytest tests/test_file.py -k test_name -v
```

### Linting / Formatting

Ruff (v0.15.9) is configured as a dev dependency. Run before committing:

```bash
# Lint all files
uv run ruff check .

# Format all files
uv run ruff format .

# Lint a single file
uv run ruff check models/registry.py

# Format a single file
uv run ruff format models/registry.py

# Auto-fix lint issues
uv run ruff check --fix .
```

## Code Style

### Imports

Order: standard library → third-party → local (`models.*`), each group separated by a blank line.
```python
import io
from os import environ

import numpy as np
import soundfile as sf
from fastapi import FastAPI

from models.base import STTModel, TTSModel
```
Prefer `from x import y` over `import x` for used symbols. Lazy-import heavy deps (e.g. `resampy`) inside functions to reduce startup cost.

### Formatting

- 4-space indentation, no tabs
- Line length: follow Ruff default (88 chars)
- Use trailing commas in multi-line lists/tuples
- Use `snake_case` for functions, variables, modules; `PascalCase` for classes; `UPPER_SNAKE_CASE` for module-level constants

### Types

- Use type hints on all function signatures and class attributes
- Import from `typing` for generics: `Optional`, `Dict`, `Type`, `Generator`
- Use built-in generics where available (Python 3.10+): `list[str]`, `tuple[np.ndarray, int]`
- Abstract base classes use `ABC` + `@abstractmethod` from `abc`

### Naming

- Model classes: descriptive noun or model name, e.g. `CohereSTT`, `KokoroTTS`
- Module-level model identifiers: `MODEL_ID = "org/name"` (string constant)
- Private registries: leading underscore, e.g. `_stt_registry`

### Error Handling

- Use `HTTPException(status_code, detail)` for API-level errors (400 for bad input, 500 for server issues)
- Include available options in error details, e.g. `f"Unknown STT model: {model!r}. Available: {list_stt_models()}"`
- Abstract methods may use `pass` as body (concrete implementations must override)
- Use `print()` for startup/loading messages (consider migrating to `logging` or `loguru` — already in dependencies)

### Architecture

- `models/base.py` — abstract `STTModel` / `TTSModel` interfaces
- `models/registry.py` — singleton registry with lazy instantiation; auto-registers on import
- `models/*.py` — concrete implementations, one per model
- `main.py` — FastAPI routes, Pydantic request models, audio I/O

### Pydantic Models

Request bodies use Pydantic `BaseModel` with sensible defaults:
```python
class SpeechRequest(BaseModel):
    input: str
    model: str = ""
    voice: str = "af_heart"
    response_format: str = "wav"
    speed: float = 1.0
```

### Audio

- STT: accepts any format `soundfile` can read; resamples to 16kHz internally
- TTS: outputs WAV, FLAC, or raw PCM (16-bit little-endian)
- Default sample rate for TTS: 24000 Hz
