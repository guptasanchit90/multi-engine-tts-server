# API Reference

The server listens on `http://0.0.0.0:8000` by default. Interactive docs (Swagger UI) are available at `http://localhost:8000/docs`.

All `/tts` requests return an `audio/mpeg` MP3 file.

---

## POST /tts — Generate Speech

The engine is selected automatically based on the `model` field.

### Request Body (JSON)

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | **required** | Text to synthesise |
| `model` | string | **required** | Model or voice identifier (see engine docs) |
| `speaker_name` | string | `null` | Named speaker — Qwen CustomVoice or Kokoro voice name |
| `voice_description` | string | `null` | Qwen VoiceDesign: voice to generate; Qwen CustomVoice: tone/emotion |
| `sample_voice_file` | string | `null` | Qwen Clone: filename (with or without `.wav`) inside `voices/` |
| `speed` | string | `"normal"` | `"slow"` · `"normal"` · `"fast"` |
| `temperature` | float | `0.0` | `0` = deterministic; `0.7` = natural variation (Qwen only) |
| `seed` | integer | auto | Fix for reproducible output; echoed in `X-Seed` response header |

### Response

- **Success:** `audio/mpeg` binary, `filename: speech.mp3`
- **Header:** `X-Seed: <integer>` — the seed actually used (useful for replaying a result)
- **Error:** JSON `{"detail": "..."}` with appropriate HTTP status code

### Model Quick Reference

| Model value | Engine | Required field |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Qwen | `speaker_name` (optional) |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | Qwen | `speaker_name` (optional) |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | Qwen | `voice_description` (required) |
| `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` | Qwen | `voice_description` (required) |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | Qwen | `sample_voice_file` (required) |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | Qwen | `sample_voice_file` (required) |
| `kokoro-v1.0` | Kokoro | `speaker_name` (optional, defaults to `af_heart`) |
| `en_US-lessac-medium` | Piper | *(none — model IS the voice)* |
| `<any piper voice stem>` | Piper | *(none)* |

### Examples

**Qwen — Custom Voice:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a custom voice.",
    "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
    "speaker_name": "Ryan",
    "voice_description": "Excited and upbeat",
    "seed": 42
  }' --output speech.mp3
```

**Qwen — Voice Design:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to the show.",
    "model": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
    "voice_description": "Deep, warm male narrator with a slight British accent",
    "speed": "slow"
  }' --output speech.mp3
```

**Qwen — Voice Cloning** (place `my_voice.wav` in `voices/` first):
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is cloned from the reference voice.",
    "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",
    "sample_voice_file": "my_voice"
  }' --output speech.mp3
```

**Kokoro:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Kokoro.",
    "model": "kokoro-v1.0",
    "speaker_name": "bf_emma",
    "speed": "normal"
  }' --output speech.mp3
```

**Piper:**
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Piper.",
    "model": "en_US-lessac-medium",
    "speed": "normal"
  }' --output speech.mp3
```

---

## GET /models — List Models

Returns all models across all engines with on-disk availability.

```bash
curl http://localhost:8000/models
```

```json
[
  {"engine": "qwen",   "model": "Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit", "mode": "custom",  "available": true},
  {"engine": "qwen",   "model": "Qwen3-TTS-12Hz-1.7B-Base-8bit",        "mode": "clone",   "available": false},
  {"engine": "kokoro", "model": "kokoro-v1.0",                           "mode": "speaker", "available": true},
  {"engine": "piper",  "model": "en_US-lessac-medium",                   "mode": "speaker", "available": true}
]
```

---

## GET /voices — List Voices

Returns all voices grouped by engine and category. Cloneable WAV files and Piper voices are read from disk at request time — no restart needed after adding new files.

```bash
curl http://localhost:8000/voices
```

```json
{
  "qwen": {
    "built_in":   ["Aiden", "Chelsie", "Dylan", "Eric", "Ethan", "Ono_Anna", "Ryan", "Serena", "Sohee", "Uncle_Fu", "Vivian"],
    "cloneable":  ["my_voice.wav"]
  },
  "kokoro": {
    "en-us": ["af_bella", "af_heart", "am_fenrir", ...],
    "en-gb": ["bf_emma", "bm_george", ...],
    "ja":    ["jf_alpha", "jm_kumo", ...],
    "zh":    ["zf_xiaobei", "zm_yunxi", ...]
  },
  "piper": {
    "en_US": ["en_US-lessac-medium"],
    "en_GB": ["en_GB-alba-medium"]
  }
}
```

---

## GET /health — Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

---

## Error Responses

All errors return JSON:

```json
{"detail": "Human-readable error message"}
```

| HTTP Status | Meaning |
|---|---|
| `422` | Invalid request (unknown model, missing required field, bad parameter) |
| `500` | Server-side error (model load failure, ffmpeg not installed, generation failed) |
