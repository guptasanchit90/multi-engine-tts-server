# Whisper MLX Engine

[OpenAI Whisper](https://github.com/openai/whisper) running locally via [MLX](https://github.com/ml-explore/mlx) on Apple Silicon. Transcribe audio to text. 5 model sizes from tiny to large-v3.

> **Apple Silicon only.** MLX needs the Metal GPU — no x86, no Docker.

---

## Models

| Model ID | HF Repo | Size | Speed |
|---|---|---|---|
| `whisper-tiny` | `mlx-community/whisper-tiny` | ~75 MB | Fastest |
| `whisper-base` | `mlx-community/whisper-base-mlx` | ~150 MB | Very fast |
| `whisper-small` | `mlx-community/whisper-small-mlx` | ~500 MB | Fast |
| `whisper-medium` | `mlx-community/whisper-medium-mlx` | ~1.5 GB | Balanced |
| `whisper-large-v3` | `mlx-community/whisper-large-v3-mlx` | ~3 GB | Best quality |

## Model Download

```bash
source venv/bin/activate
pip install -U huggingface_hub

# Pick your size:
hf download mlx-community/whisper-base-mlx --local-dir models/whisper/whisper-base
```

Expected layout:

```
models/
└── whisper/
    └── whisper-base/
        ├── config.json
        ├── model.safetensors
        └── ...
```

## Try it

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@speech.mp3" \
  -F "model=whisper-base" \
  -F "language=en"
```

```json
{"text": "Hello world, this was transcribed locally."}
```

### Response formats

| Format | Returns |
|---|---|
| `json` (default) | `{"text": "..."}` |
| `text` | Plain text body |
| `verbose_json` | Full result with segments, timestamps, language |

## Parameters

| Field | Required | Notes |
|---|---|---|
| `model` | **Yes** | One of `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3` |
| `language` | No | Language hint — auto-detected if omitted |
| `temperature` | No | Sampling temperature (default `0.0` for deterministic) |
| `response_format` | No | `json` · `text` · `verbose_json` (default `json`) |

### Header options

| Header | Effect |
|---|---|
| `x-save-output: true` | Saves transcription result as JSON in `outputs/server/` |

## Limitations

- Apple Silicon only — no x86, no Docker
- First load downloads model weights if not cached locally
- Large models (medium, large-v3) need ~2–6 GB RAM

## Troubleshooting

| Problem | Fix |
|---|---|
| `mlx_whisper not found` | Run `pip install mlx-whisper` |
| Model not found | Download it — see model download above |
| Poor transcription quality | Use a larger model (`whisper-medium` or `whisper-large-v3`) |
| Wrong language detected | Pass `language` parameter explicitly |
| ffmpeg failed | Run `brew install ffmpeg` |
