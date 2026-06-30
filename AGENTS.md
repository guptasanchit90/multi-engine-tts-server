# AGENTS.md — Sonus

Guidelines for AI coding agents working in this repository.

## Key files

```
server.py                   # FastAPI entry point, routing, OpenAI-compatible endpoint
src/
  utils.py                  # Shared constants + helpers
  cache.py                  # ModelCache[T] with TTL eviction for MLX engines
  audio.py                  # WAV silence trimming + MP3 conversion
  engines/
    base.py                 # TTSEngine protocol + @register + BaseEngine
    qwen.py / chatterbox.py # MLX-based engines (Apple Silicon)
    kokoro.py / piper.py    # ONNX-based engines
    __init__.py             # Imports all engine modules
static/                     # Web UI (vanilla JS, no build)
docs/                       # API reference + per-engine docs
notebooks/                  # Quickstart notebooks (Colab, Kaggle, SageMaker)
```

## Setup

```bash
source venv/bin/activate          # always use project venv
pip install -r requirements.txt
brew install ffmpeg
python server.py                  # http://0.0.0.0:8000
```

## Lint / typecheck

```bash
ruff check server.py src/
ruff format server.py src/
pyright server.py src/
```

## Adding an engine

Create `src/engines/<name>.py` implementing `TTSEngine` (5 methods: `claims`, `list_models`, `list_voices`, `validate`, `generate`), add `@register`, import in `src/engines/__init__.py`, and write `docs/engines/<name>.md`. See `CONTRIBUTING.md` for full details.

## Notebooks

Quickstart notebooks in `notebooks/` mirror the onboarding flow. When adding/renaming engines, models, dependencies, or endpoints, keep the notebooks in sync — update model lists, install commands, API calls, and UI references.

## Dependencies

`requirements.txt` has all deps (including MLX). `requirements-non-mlx.txt` mirrors it without Apple Silicon packages. Keep both in sync — when adding/updating a dependency in `requirements.txt`, add it to `requirements-non-mlx.txt` too (unless it's MLX-specific).

## Engine patterns

- Use `ModelCache` for MLX models (`src/cache.py`)
- Call `clean_memory()` from `src.utils` after releasing models
- Import shared utils (`resolve_voice`, `model_path`, `convert_to_wav_24k`, `scan_wav_voices`) from `src.utils`
- Use `os.path.join` for paths, never hardcode
- All config in `CONTRIBUTING.md` and `docs/`
