# Contributing

Contributions are welcome — bug reports, new engines, documentation improvements, and pull requests.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Overview](#project-overview)
- [Adding a New Engine](#adding-a-new-engine)
- [Code Style](#code-style)
- [OpenAI-Compatible Endpoint](#openai-compatible-endpoint)
- [Output File Conventions](#output-file-conventions)
- [Key Dependencies](#key-dependencies)
- [Reporting Bugs](#reporting-bugs)
- [Pull Requests](#pull-requests)

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/local-tts-server.git
cd local-tts-server

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg
```

Download at least one engine's models (see the engine docs in `docs/engines/`) before running.

---

## Project Overview

A FastAPI HTTP server exposing a multi-engine TTS API, running locally on Apple Silicon.
Engines live in `src/engines/` and implement the `TTSEngine` protocol (`src/engines/base.py`).

### Architecture

```
server.py              # FastAPI app, routing, entry point
src/
  utils.py             # Shared constants + helpers (voice resolution, ffmpeg, memory)
  cache.py             # Generic ModelCache[T] with TTL eviction for MLX engines
  audio.py             # WAV silence trimming + MP3 conversion
  engines/
    base.py            # TTSEngine protocol + @register decorator + BaseEngine mixin
    qwen.py            # Qwen3 via mlx-audio (Apple Silicon)
    chatterbox.py      # Chatterbox Turbo via mlx-audio (Apple Silicon)
    kokoro.py          # Kokoro-82M via kokoro-onnx (ONNX)
    piper.py           # Piper via piper-tts (ONNX)
    __init__.py        # Imports all engine modules to trigger @register decorators
static/
  index.html           # Web UI form for all TTS parameters
  app.js               # Web UI logic — form submission, audio playback
models/                # Downloaded model files (gitignored)
voices/                # Cloneable WAV samples (gitignored except .gitkeep)
outputs/               # Generated audio (gitignored)
docs/                  # API and per-engine documentation
```

---

## Adding a New Engine

Engines are auto-discovered via the `@register` decorator — **no `server.py` edits needed**.

### Registration steps

| Step | Action |
|---|---|
| 1 | Create `src/engines/<name>.py` implementing `TTSEngine` |
| 2 | Add `@register` decorator from `src.engines.base` |
| 3 | Set `engine_name` property (used by `/voices` route) |
| 4 | Import shared utilities from `src.utils` instead of re-implementing them |
| 5 | For MLX engines — use `ModelCache` from `src.cache` for model caching |
| 6 | Register the module in `src/engines/__init__.py` by importing the engine class |
| 7 | Add `docs/engines/<name>.md` |

### Quick-start template

```python
# src/engines/myengine.py
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

from src.utils import VOICES_DIR, SAMPLE_RATE, resolve_voice, clean_memory
from .base import BaseEngine, register

MODELS_DIR = os.path.join(os.getcwd(), "models", "myengine")
MODEL_ID = "my-voice-model"


@register
class MyEngine(BaseEngine):
    @property
    def engine_name(self) -> str:
        return "myengine"

    def claims(self, model: str) -> bool:
        return model == MODEL_ID

    def list_models(self) -> list[dict]:
        return [{
            "engine": "myengine",
            "model": MODEL_ID,
            "mode": "speaker",
            "available": os.path.exists(MODELS_DIR),
        }]

    def list_voices(self) -> dict:
        return super().list_voices()

    def validate(self, request: dict) -> None:
        _model = request["model"]
        if _model != MODEL_ID:
            raise HTTPException(status_code=422, detail=f"Unknown model '{_model}'")

    def generate(self, request: dict, tmp_dir: str) -> str:
        text = request["text"]
        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        # ... write audio to wav_path ...
        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="TTS produced no output file")
        return wav_path
```

### The five required methods

| Method | Responsibility |
|---|---|
| `claims(model)` | Return `True` if this engine owns the model identifier |
| `list_models()` | Return metadata dicts with `engine`, `model`, `mode`, `available` keys |
| `list_voices()` | Return `{category: [voice_id, ...]}` or `{}` |
| `validate(request)` | Raise `HTTPException(422)` on bad input — called before any model loading |
| `generate(request, tmp_dir)` | Run inference, write `audio_000.wav` into `tmp_dir`, return its path |

### Shared utilities at `src/utils.py`

| Function | What it does |
|---|---|
| `resolve_voice(filename)` | Find a `.wav` in `voices/` by name or full filename |
| `model_path(base_dir, folder)` | Resolve model folder path (handles HF snapshot layout) |
| `get_audio_duration(filepath)` | Get audio duration via ffprobe |
| `convert_to_wav_24k(in, out)` | Re-encode any audio to 24kHz mono WAV via ffmpeg |
| `clean_memory()` | `gc.collect()` + `mx.clear_cache()` if MLX available |
| `scan_wav_voices(dir)` | List `.wav` files in a directory (sorted, no dotfiles) |

Shared constants: `MODELS_DIR` (base), `VOICES_DIR`, `SAMPLE_RATE`.

### Model cache for MLX engines

Engines using `mlx` should cache their loaded model with `ModelCache` from `src.cache`:

```python
from src.cache import ModelCache

_model_cache = ModelCache(ttl=10, tag="myengine")

def generate(self, request, tmp_dir):
    model = _model_cache.get_or_load(
        request["model"],
        lambda: load_model(Path(model_path)),
    )
    # ... use model ...
```

### Engine file structure (convention)

1. Stdlib imports (alphabetical)
2. Warning suppression
3. `try/except ImportError` for optional platform deps (MLX) — set `_MLX_AVAILABLE = False`
4. `try/except ImportError` for hard deps (fastapi)
5. Shared imports from `src.utils`, `src.cache`
6. Engine-specific imports from `.base`
7. Constants (`MODELS_DIR`, `MODEL_ID`, engine-specific maps)
8. Engine helpers (if any, prefixed with `_`)
9. `@register` class implementing `TTSEngine`

---

## Code Style

### General

- **Engines belong in `src/engines/`.** Shared utilities in `src/utils.py`, `src/cache.py`, `src/audio.py`.
- **No classes outside engine files.** Procedural helpers are preferred everywhere else.
- **Line length:** keep under ~100 characters. Match existing style.
- **No trailing whitespace.**

### Imports

Group imports in this order, separated by blank lines:

1. Stdlib (alphabetical)
2. Warning suppression
3. Third-party `try/except ImportError` (optional deps like `mlx`)
4. Third-party `try/except ImportError` (hard deps like `fastapi`)
5. Project modules (`src.utils`, `src.cache`)
6. Relative engine imports (`.base`)

```python
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    _MLX_AVAILABLE = True
except ImportError:
    pass

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install -r requirements.txt") from exc

from src.cache import ModelCache
from src.utils import VOICES_DIR, SAMPLE_RATE, resolve_voice, clean_memory
from .base import BaseEngine, register
```

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
| Engine property | `engine_name` (property) | `return "qwen"` |

### Type Annotations

Engine files use type annotations — maintain them when editing engine code:

```python
def claims(self, model: str) -> bool: ...
def list_models(self) -> list[dict]: ...
```

- Use `str | None` over `Optional[str]` (Python 3.10+ union syntax).
- `server.py` and helpers have no annotations — adding them is welcome but not required.
- If you annotate, annotate all parameters and the return type consistently.

### Constants

Per-engine constants live in the engine file. Shared constants are in `src/utils.py`:

```python
# src/utils.py (shared)
MODELS_DIR  = os.path.join(os.getcwd(), "models")
VOICES_DIR  = os.path.join(os.getcwd(), "voices")
SAMPLE_RATE = 24000

# Engine file (engine-specific)
MODELS_DIR = os.path.join(os.getcwd(), "models", "myengine")
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

Use `check=True` when failure matters (ffmpeg conversion). Use `check=False` for optional commands (`afplay`).

### Memory Management

Call `clean_memory()` from `src.utils` (handles `gc.collect()` + `mx.clear_cache()`) after releasing heavy MLX models:

```python
finally:
    del model
    clean_memory()
```

The `ModelCache` class (`src/cache.py`) calls `clean_memory()` automatically on eviction.

### Warning Suppression

Keep at the top of every engine file:

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
```

`os.environ["TOKENIZERS_PARALLELISM"] = "false"` is set once in `server.py`.

---

## OpenAI-Compatible Endpoint

The server exposes `POST /v1/audio/speech` following the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech) format.

### Model aliases

The model manifest is built **dynamically** from each engine's `list_models()` at request time. Each engine declares its aliases directly in `list_models()`:

```python
def list_models(self) -> list[dict]:
    return [{
        "id": "kokoro",
        "name": "Kokoro 82M",
        "engine": "kokoro",
        "model": "kokoro-v1.0",
        "mode": "speaker",
        "capabilities": ["speaker", "voice_blend"],
        "description": "...",
        "available": True,
    }]
```

For catch-all engines (like Piper, where voice files are the models), use `"model": ""`.

### Voice field mapping

| Capability | `voice` param maps to |
|---|---|
| `speaker` / `voice_blend` | `speaker_name` |
| `voice_prompt` | `voice_description` |
| `voice_clone` | `sample_voice_file` |

### Adding a new alias

Add a new entry to the engine's `list_models()` return dict:
1. Pick a short, descriptive `id` (unique across all engines)
2. Set `engine` to match `engine_name`
3. Set `model` to the internal model identifier
4. Declare `capabilities`
5. Set `available` based on model file presence

### OpenAI standard name mapping

`OPENAI_MODEL_ALIASES` in `server.py` maps standard OpenAI names (`tts-1`, `tts-1-hd`) to aliases from the dynamic manifest.

---

## Output File Conventions

- Generated MP3s are saved to `outputs/server/<uuid>.mp3` by the server.
- Temp work directories are `<tempdir>/tts_<uuid>/` and deleted after each request.
- Cloneable voice samples live in `voices/*.wav` (gitignored except `.gitkeep`).
- Piper models: `models/piper/<name>.onnx` + `<name>.onnx.json`.
- Kokoro models: `models/kokoro/kokoro-v1.0.onnx` + `voices-v1.0.bin`.
- Qwen models: `models/qwen/<folder>/` (HuggingFace snapshot layout).
- Chatterbox models: loaded directly from HF cache (`mlx-community/Chatterbox-Turbo-TTS-fp16`).

## Chatterbox Turbo Model Setup

```bash
source venv/bin/activate
hf download mlx-community/Chatterbox-Turbo-TTS-fp16
```

The fp16 variant (~1.2 GB) is cached in the HuggingFace cache and loaded directly by mlx-audio. The S3TokenizerV2 is auto-downloaded by mlx-audio on first load.

---

## Key Dependencies

| Package | Used by | Purpose |
|---|---|---|
| `fastapi`, `uvicorn` | `server.py` | HTTP server |
| `mlx`, `mlx-audio`, `mlx-metal` | `qwen.py`, `chatterbox.py` | Apple Silicon inference |
| `kokoro-onnx` | `kokoro.py` | Kokoro ONNX runtime |
| `piper-tts` | `piper.py` | Piper ONNX runtime |
| `numpy`, `soundfile` | `kokoro.py` | Audio array I/O |
| `transformers`, `tokenizers` | `mlx-audio` (transitive) | Tokenisation |
| `huggingface_hub` | `mlx-audio` (transitive) | Model downloading |

Pin new direct dependencies to exact versions (`==`) in `requirements.txt`. Transitive deps are installed automatically — do not pin them explicitly unless resolving a conflict.

---

## Reporting Bugs

Open an issue with:
- macOS version and Apple Silicon chip (e.g. M2 Pro)
- Python version (`python3 --version`)
- The exact command or request that failed
- Full error output

---

## Pull Requests

- Keep changes focused — one logical change per PR.
- Update the relevant `docs/` file if you change behaviour.
- Test with at least one engine before submitting.
