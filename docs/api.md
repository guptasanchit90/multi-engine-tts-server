# API Reference

The server lives at `http://0.0.0.0:8000`. Hit it with curl, Python, or whatever speaks HTTP.

- Interactive docs (Swagger UI): `http://localhost:8000/api-docs`
- Web UI: `http://localhost:8000/` — a dark-themed form that lets you play with everything

All `/tts` requests return an MP3 file (`audio/mpeg`). Ping. Done.

---

## POST /tts — Make some noise

The server picks which engine to use based on the `model` field you send. Simple.

### Request body

| Field | Type | Default | What it does |
|---|---|---|---|
| `text` | string | **required** | The words you want spoken |
| `model` | string | **required** | Which model/voice (see engine docs) |
| `speaker_name` | string | `null` | A named speaker — Qwen CustomVoice or Kokoro voice ID |
| `voice_description` | string | `null` | Qwen VoiceDesign: describe the voice; Qwen CustomVoice: set tone/emotion |
| `sample_voice_file` | string | `null` | Qwen Clone: WAV filename (`.wav` optional) inside `voices/` |
| `speed` | string | `"normal"` | `"slow"` · `"normal"` · `"fast"` |
| `temperature` | float | `0.0` | `0` = same result every time; `0.7` = roll the dice (Qwen only) |
| `seed` | integer | auto | Fix the randomness. Echoed back in `X-Seed` header. |
| `add_pauses` | boolean | `true` | Breathe at punctuation (`.` `,` `?` `!` — you get the idea) |

### Response

- **Win:** `audio/mpeg` binary, filename `speech.mp3`
- **Header:** `X-Seed: <integer>` — useful if you want to replay the same output
- **Lose:** JSON `{"detail": "..."}` with an HTTP error code

### Model cheat sheet

| Model value | Engine | You need this field |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Qwen | `speaker_name` (optional) |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit` | Qwen | `speaker_name` (optional) |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` | Qwen | `voice_description` (required) |
| `Qwen3-TTS-12Hz-0.6B-VoiceDesign-8bit` | Qwen | `voice_description` (required) |
| `Qwen3-TTS-12Hz-1.7B-Base-8bit` | Qwen | `sample_voice_file` (required) |
| `Qwen3-TTS-12Hz-0.6B-Base-8bit` | Qwen | `sample_voice_file` (required) |
| `kokoro-v1.0` | Kokoro | `speaker_name` (optional, defaults to `af_heart`) |
| `en_US-lessac-medium` | Piper | *(none — the model IS the voice)* |
| `<any piper voice stem>` | Piper | *(none)* |
| `chatterbox-turbo-fp16` | Chatterbox | `sample_voice_file` (required) |

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

**Qwen — Voice Cloning** (drop `my_voice.wav` into `voices/` first):
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

## GET /models — What's installed?

Returns every model, across all engines, with an `available` flag so you know what's ready.

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

## GET /voices — Who can speak?

All voices, grouped by engine and category. Add WAV files to `voices/` and they show up instantly — no restart needed.

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

## GET /v1/models — OpenAI-style model list

Returns friendly aliases and capabilities for the OpenAI-compatible endpoint.

```bash
curl http://localhost:8000/v1/models
```

```json
[
  {"id": "kokoro", "name": "Kokoro 82M", "engine": "kokoro", "capabilities": ["speaker", "voice_blend"], "available": true, ...},
  {"id": "qwen-voice", "name": "Qwen3 Pro Voice Design", "engine": "qwen", "capabilities": ["voice_prompt"], "available": true, ...},
  {"id": "qwen-clone", "name": "Qwen3 Pro Voice Clone", "engine": "qwen", "capabilities": ["voice_clone"], "available": true, ...}
]
```

---

## POST /v1/audio/speech — OpenAI-compatible TTS

Drop-in replacement for [OpenAI's TTS endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech). Point your existing code at `http://localhost:8000` and it just works.

### Request body

| Field | Type | Default | What it does |
|---|---|---|---|
| `model` | string | **required** | Alias from `/v1/models` (e.g. `kokoro`, `qwen-voice`, `qwen-clone`) |
| `input` | string | **required** | Text to speak (max 4096 chars) |
| `voice` | string | `null` | Maps to `speaker_name`, `voice_description`, or `sample_voice_file` |
| `response_format` | string | `"mp3"` | `mp3` · `wav` · `pcm` |
| `speed` | number | `1.0` | `0.25` – `4.0` |

### How `voice` maps

| Model capability | `voice` becomes |
|---|---|
| `speaker` / `voice_blend` | Speaker name (e.g. `af_heart`, `Vivian`) |
| `voice_prompt` | Voice description (e.g. `"A warm, deep voice"`) |
| `voice_clone` | WAV filename in `voices/` (e.g. `my_voice.wav`) |

### Response

- **Win:** audio file with `Content-Type: audio/mpeg` (or `audio/wav`, `audio/L16`)
- **Header:** `X-Seed: <integer>` — the seed used
- **Lose:** JSON `{"detail": "..."}`

### Example

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kokoro",
    "input": "Hello from the OpenAI-compatible endpoint!",
    "voice": "af_bella",
    "speed": 1.0
  }' --output speech.mp3
```

---

## DELETE /outputs — Clean up

Deletes all generated MP3s from `outputs/server/`. Returns what it nuked.

```bash
curl -X DELETE http://localhost:8000/outputs
```

```json
{"deleted": 5, "files": ["abc123.mp3", "def456.mp3", ...]}
```

---

## GET /health — Is it alive?

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

Yep.

---

## Errors

All errors come back as JSON:

```json
{"detail": "Something went wrong — here's what"}
```

| HTTP status | What it means |
|---|---|
| `422` | Bad request — unknown model, missing field, bad value |
| `500` | Server oops — model didn't load, ffmpeg not found, generation bombed |
