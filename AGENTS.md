# AGENTS.md — Qwen3-TTS Apple Silicon

Guidelines for AI coding agents working in this repository.

---

## Project Overview

A FastAPI HTTP server (`server.py`) exposing a multi-engine TTS API, running locally on Apple Silicon.
Engines live in `src/engines/` and implement a shared `TTSEngine` protocol (`src/engines/base.py`).
macOS-only — depends on the MLX Metal backend and `ffmpeg`.

```
server.py              # FastAPI app, routing, WAV→MP3, entry point
src/
  engines/
    base.py            # TTSEngine protocol (Protocol class)
    qwen.py            # Qwen3 via mlx-audio (Apple Silicon)
    kokoro.py          # Kokoro-82M via kokoro-onnx (ONNX)
    piper.py           # Piper via piper-tts (ONNX)
models/                # Downloaded model files (gitignored)
voices/                # Cloneable WAV samples (gitignored except .gitkeep)
outputs/               # Generated audio (gitignored)
docs/                  # API and per-engine documentation
```

---

## Environment Setup

```bash
python3.13 -m venv venv          # Python 3.13+ required — see note below
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

- **Python version:** 3.13+ required (`brew install python@3.13`). Older versions (e.g. macOS
  system Python 3.9) will fail to resolve pinned package versions — the error looks like
  `Could not find a version that satisfies the requirement fastapi==x.y.z`.
- **Platform:** Apple Silicon macOS only (MLX Metal backend).
- **Always use the project venv** — do not use system `pip3` or `python3` directly.
- Models are downloaded separately and stored in `models/` (gitignored).

---

## Running the Application

```bash
source venv/bin/activate
python server.py          # listens on http://0.0.0.0:8000
```

API docs auto-generated at `http://localhost:8000/docs`.

---

## Build / Lint / Test Commands

There is no build system, linter config, or test suite. No `Makefile`, `pyproject.toml`, or CI.

**If introducing tooling, use these conventions:**

```bash
# Lint / format
ruff check server.py src/
ruff format server.py src/

# Type checking
pyright server.py src/

# Run a single test
venv/bin/pytest tests/test_engines.py::test_function_name -v

# Run all tests
venv/bin/pytest tests/ -v
```

Place new tests in `tests/` using `pytest`. Always invoke via `venv/bin/pytest` to avoid
accidentally using a system interpreter.

---

## Adding a New Engine

1. Create `src/engines/<name>.py` implementing all five methods of `TTSEngine` (see `src/engines/base.py`).
2. Import and append an instance to `ENGINES` in `server.py`.
3. Add a `docs/engine-<name>.md` and update `docs/api.md`.

The five required methods:

| Method | Responsibility |
|---|---|
| `claims(model)` | Return `True` if this engine owns the model identifier |
| `list_models()` | Return metadata dicts with `engine`, `model`, `mode`, `available` keys |
| `list_voices()` | Return `{category: [voice_id, ...]}` or `{}` |
| `validate(request)` | Raise `HTTPException(422)` on bad input — called before any model loading |
| `generate(request, tmp_dir)` | Run inference, write `audio_000.wav` into `tmp_dir`, return its path |

---

## Code Style Guidelines

### General

- **`server.py` stays single-file.** Engines belong in `src/engines/`. Do not create new modules
  unless explicitly asked.
- **No classes outside engine files.** Procedural helpers are preferred everywhere else.
- **Line length:** keep under ~100 characters. Match existing style.
- **No trailing whitespace.**

### Imports

Order: stdlib (alphabetical) → third-party. Wrap third-party imports in `try/except ImportError`:

```python
# stdlib
import gc
import os
import subprocess

# third-party
try:
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model
except ImportError:
    pass   # engine guards against _MLX_AVAILABLE = False at call site
```

- Use `try/except ImportError` with a user-friendly message and `sys.exit(1)` for hard dependencies.
- Use `pass` (and a module-level flag like `_MLX_AVAILABLE`) for optional platform dependencies.
- No wildcard imports (`from foo import *`).
- Lazy imports inside a function are acceptable for platform-specific modules (`termios`).

### Naming Conventions

| Kind | Convention | Example |
|---|---|---|
| Functions | `snake_case` verb_noun | `load_model`, `scan_voices`, `clean_memory` |
| Constants | `UPPER_SNAKE_CASE` | `MODELS_DIR`, `SAMPLE_RATE`, `MODEL_ID` |
| Private helpers | `_leading_underscore` | `_model_path`, `_resolve_voice`, `_lang_for_voice` |
| Local variables | `snake_case` | `tmp_dir`, `wav_path`, `speed_key` |
| Boolean flags | Positive prefix | `_MLX_AVAILABLE`, `is_valid` |

### Type Annotations

Engine files use type annotations — maintain them when editing engine code:

```python
def _model_path(folder_name: str) -> str | None: ...
def claims(self, model: str) -> bool: ...
def list_models(self) -> list[dict]: ...
```

- Use `str | None` over `Optional[str]` (Python 3.10+ union syntax).
- `server.py` and helpers have no annotations — adding them is welcome but not required.
- If you annotate, annotate all parameters and the return type consistently.

### Constants

Define at module level near the top of each file:

```python
MODELS_DIR  = os.path.join(os.getcwd(), "models", "qwen")
VOICES_DIR  = os.path.join(os.getcwd(), "voices")
SAMPLE_RATE = 24000
```

Use `os.getcwd()` — never hardcode absolute paths.

### Error Handling

**In engines** — raise `HTTPException` so FastAPI returns a clean JSON error:

```python
raise HTTPException(status_code=422, detail="'voice_description' is required")
raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
```

**Guard clauses** — return `None` early rather than deep nesting:

```python
def _model_path(folder_name: str) -> str | None:
    full = os.path.join(MODELS_DIR, folder_name)
    if not os.path.exists(full):
        return None
    return full
```

**Silent suppression** — for optional features only:

```python
try:
    import termios
    termios.tcflush(sys.stdin, termios.TCIOFLUSH)
except (ImportError, OSError):
    pass
```

- Never use bare `except:` — always catch at minimum `Exception`.
- Do not re-raise unless there is a specific reason to propagate.
- Always clean up resources in `finally` (models, temp dirs).

### Path Handling

- Use `os.path.join` for all path construction — never string concatenation.
- Temp directories: `tempfile.gettempdir()` + a unique suffix. Clean up with `shutil.rmtree`.

### Subprocess Usage

Use `subprocess.run` with explicit stdout/stderr suppression:

```python
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-i", src, dst],
    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
)
```

Use `check=True` when failure matters (ffmpeg conversion). Use `check=False` for optional
commands (`afplay`).

### Memory Management

Call `gc.collect()` after releasing heavy MLX models:

```python
finally:
    del model
    gc.collect()
```

### Warning Suppression

Keep at the top of every engine file and `server.py`:

```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
```

---

## Output File Conventions

- Generated MP3s are saved to `outputs/server/<uuid>.mp3` by the server.
- Temp work directories are `<tempdir>/tts_<uuid>/` and deleted after each request.
- Cloneable voice samples live in `voices/*.wav` (gitignored except `.gitkeep`).
- Piper models: `models/piper/<name>.onnx` + `<name>.onnx.json`.
- Kokoro models: `models/kokoro/kokoro-v1.0.onnx` + `voices-v1.0.bin`.
- Qwen models: `models/qwen/<folder>/` (HuggingFace snapshot layout).

---

## Key Dependencies

| Package | Used by | Purpose |
|---|---|---|
| `fastapi`, `uvicorn` | `server.py` | HTTP server |
| `mlx`, `mlx-audio`, `mlx-metal` | `qwen.py` | Apple Silicon inference |
| `mlx-lm` | `mlx-audio` (transitive) | LM utilities |
| `kokoro-onnx` | `kokoro.py` | Kokoro ONNX runtime |
| `piper-tts` | `piper.py` | Piper ONNX runtime |
| `numpy`, `soundfile` | `kokoro.py` | Audio array I/O |
| `transformers`, `tokenizers` | `mlx-audio` (transitive) | Tokenisation |
| `huggingface_hub` | `mlx-audio` (transitive) | Model downloading |

Pin new direct dependencies to exact versions (`==`) in `requirements.txt`. Transitive deps
are installed automatically — do not pin them explicitly unless resolving a conflict.
