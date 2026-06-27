# Development Setup

## Prerequisites

- macOS with Apple Silicon (M1–M4)
- Python 3.13+ (`brew install python@3.13`)
- [ffmpeg](https://ffmpeg.org/) (`brew install ffmpeg`)

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/local-tts-server.git
cd local-tts-server

python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Always use the project venv** — do not use system `pip3` or `python3` directly.

## Running

```bash
source venv/bin/activate
python server.py          # listens on http://0.0.0.0:8000
```

- Interactive API docs: `http://localhost:8000/docs`
- Web UI: `http://localhost:8000/`
- OpenAI-compatible endpoint: `http://localhost:8000/v1/audio/speech`

## Linting & Type Checking

```bash
ruff check server.py src/
ruff format server.py src/
pyright server.py src/
```

## Testing

```bash
venv/bin/pytest tests/ -v
```

Place new tests in `tests/` using `pytest`.

## Model Download

Each engine requires model files. See the individual engine docs in `docs/engines/` for download instructions.

### Docker

Kokoro and Piper engines can run in Docker:

```bash
docker build -t tts-server .
docker run -p 8000:8000 \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/voices:/app/voices" \
  -v "$(pwd)/outputs:/app/outputs" \
  tts-server
```

Qwen and Chatterbox require Apple Silicon Metal GPU and cannot run in Docker.
