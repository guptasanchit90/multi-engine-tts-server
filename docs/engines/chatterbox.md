# Chatterbox Turbo Engine

[Chatterbox-Turbo-TTS](https://huggingface.co/mlx-community/Chatterbox-Turbo-TTS-fp16) via MLX on Apple Silicon. One job, done well: high-quality voice cloning from a reference WAV.

> **Apple Silicon only.** MLX needs the Metal GPU — no x86, no Docker.

---

## Model Download

Grab the MLX-converted fp16 weights:

```bash
source venv/bin/activate
hf download mlx-community/Chatterbox-Turbo-TTS-fp16
```

The model (~1.2 GB) is cached by HuggingFace. S3TokenizerV2 auto-downloads on first load.

For manual placement:

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

## Try it

Drop a WAV in `voices/` first, then:

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
| `sample_voice_file` | **Yes** | WAV filename in `voices/` (`.wav` optional) |
| `temperature` | No | `0.0` = same every time; `0.7` = variation |
| `seed` | No | Fix for reproducible output |

### Reference audio requirements

- Minimum **5 seconds** — shorter clips get rejected
- Any format ffmpeg can read (auto-converted to 24kHz mono)
- Cleaner recordings = better clones

---

## Limitations

- Voice cloning only — no built-in speakers, no voice design
- English only
- Apple Silicon only

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `mlx_audio not found` | Run `source venv/bin/activate` |
| Model not found | Check `GET /models` after downloading |
| Poor clone quality | Cleaner reference WAV, at least 5 seconds |
| ffmpeg failed | Run `brew install ffmpeg` |
