# Chatterbox Engines

[Chatterbox](https://huggingface.co/ResembleAI/chatterbox) and [Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) via MLX on Apple Silicon. Voice cloning from a reference WAV.

> **Apple Silicon only.** MLX needs the Metal GPU — no x86, no Docker.

---

## Available Models

| Model key | Family | Size | Languages | Extra features |
|---|---|---|---|---|
| `chatterbox-turbo-fp16` | Turbo | ~1.2 GB | English | — |
| `chatterbox-turbo-4bit` | Turbo | 812 MB | English | Smaller, faster |
| `chatterbox-fp16` | Regular | 2.7 GB | **16 languages** | Emotion control (`exaggeration`) |
| `chatterbox-8bit` | Regular | 1.28 GB | **16 languages** | Emotion control, lighter |

## Model Download

**Turbo:**
```bash
hf download mlx-community/Chatterbox-Turbo-TTS-fp16   --local-dir models/chatterbox/Chatterbox-Turbo-TTS-fp16
hf download mlx-community/Chatterbox-Turbo-TTS-4bit    --local-dir models/chatterbox/Chatterbox-Turbo-TTS-4bit
```

**Regular (multilingual):**
```bash
hf download mlx-community/Chatterbox-TTS-fp16  --local-dir models/chatterbox/Chatterbox-TTS-fp16
hf download mlx-community/Chatterbox-TTS-8bit   --local-dir models/chatterbox/Chatterbox-TTS-8bit
```

**S3Tokenizer (required by all Chatterbox models):**
```bash
hf download mlx-community/S3TokenizerV2 --local-dir models/chatterbox/s3_tokenizer
```

Expected layout:
```
models/
└── chatterbox/
    ├── Chatterbox-Turbo-TTS-fp16/
    ├── Chatterbox-Turbo-TTS-4bit/
    ├── Chatterbox-TTS-fp16/
    ├── Chatterbox-TTS-8bit/
    └── s3_tokenizer/              ← S3Tokenizer weights (~495 MB)
```

---

## Try it

Drop a WAV in `voices/` first, then:

**Turbo (English):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox-clone",
    "input": "This is Chatterbox Turbo voice cloning.",
    "voice": "my_voice.wav"
  }' \
  --output speech.mp3
```

**Regular (multilingual + emotion):**
```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chatterbox-clone",
    "input": "Olá, tudo bem? Este é um teste de clonagem de voz.",
    "voice": "my_voice.wav"
  }' \
  --output speech.mp3
```

---

## Parameters

| Field | Required | Applies to | Notes |
|---|---|---|---|
| `voice` | **Yes** | All | WAV filename in `voices/` (`.wav` optional) |

### Supported languages (regular Chatterbox)

`en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`, `ru`, `nl`, `cs`, `ar`, `zh`, `ja`, `hu`, `ko`

### Reference audio requirements

- Minimum **5 seconds** — shorter clips get rejected
- Any format ffmpeg can read (auto-converted to 24kHz mono)
- Cleaner recordings = better clones

---

## Limitations

- Voice cloning only — no built-in speakers, no voice design
- Turbo models: English only
- Apple Silicon only

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `mlx_audio not found` | Run `source venv/bin/activate` |
| Model not found | Check `GET /v1/models` after downloading |
| Poor clone quality | Cleaner reference WAV, at least 5 seconds |
| Speech truncates mid-sentence | `cfg_weight` too high for quantized models — keep at `0.0` (default) |
| ffmpeg failed | Run `brew install ffmpeg` |
