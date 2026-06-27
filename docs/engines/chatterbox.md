# Chatterbox Turbo Engine

The Chatterbox Turbo engine runs [Chatterbox-Turbo-TTS](https://huggingface.co/mlx-community/Chatterbox-Turbo-TTS-fp16) via MLX on Apple Silicon. It provides high-quality voice cloning from a reference WAV file.

> **Apple Silicon only.** MLX requires the Metal GPU backend and cannot run on x86 or inside Docker.

---

## Model Download

Download the MLX-converted fp16 weights using the HuggingFace Hub (with venv active):

```bash
source venv/bin/activate
hf download mlx-community/Chatterbox-Turbo-TTS-fp16
```

The model (~1.2 GB) is cached in the HuggingFace cache and loaded directly by mlx-audio. The S3TokenizerV2 is auto-downloaded on first load.

For manual placement, the model can also be downloaded to `models/chatterbox/`:

```bash
hf download mlx-community/Chatterbox-Turbo-TTS-fp16 \
  --local-dir models/chatterbox/Chatterbox-Turbo-TTS-fp16
```

Expected layout:

```
models/
└── chatterbox/
    └── Chatterbox-Turbo-TTS-fp16/
```

---

## Example Request

Chatterbox requires a reference WAV file for voice cloning. Place a `.wav` file in `voices/` first:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is Chatterbox Turbo voice cloning.",
    "model": "chatterbox-turbo-fp16",
    "sample_voice_file": "my_voice.wav"
  }' \
  --output speech.mp3
```

---

## Parameters

| Field | Required | Notes |
|---|---|---|
| `sample_voice_file` | **Yes** | WAV filename in `voices/` with or without `.wav` extension |
| `temperature` | No | `0.0` = deterministic (default); `0.7` = natural variation |
| `seed` | No | Fix for reproducible output |

### Reference Audio Requirements

- Minimum duration: **5 seconds** — shorter clips will be rejected
- Format: any format ffmpeg can read (auto-converted to 24kHz mono WAV)
- Quality: clearer recordings produce better clones

---

## Limitations

- Voice cloning only — no built-in speakers or voice design
- English language only
- Requires Apple Silicon (MLX Metal backend)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `mlx_audio not found` | Run `source venv/bin/activate` |
| `Model not found` | Check `GET /models` after downloading |
| Poor cloning quality | Provide a cleaner reference WAV (minimum 5 seconds) |
| `ffmpeg` conversion failed | Run `brew install ffmpeg` |
