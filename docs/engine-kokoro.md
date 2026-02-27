# Kokoro Engine

The Kokoro engine runs [Kokoro-82M](https://github.com/thewh1teagle/kokoro-onnx) via ONNX Runtime. It supports 54 built-in voices across 9 languages with near real-time performance on Apple Silicon.

> Kokoro uses a deterministic ONNX graph — output is reproducible for the same input without needing a fixed seed.

---

## Model Download

Download both files and place them in `models/kokoro/`:

```bash
mkdir -p models/kokoro
cd models/kokoro
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Expected layout:

```
models/
└── kokoro/
    ├── kokoro-v1.0.onnx    (~300 MB)
    └── voices-v1.0.bin
```

Use `"kokoro-v1.0"` as the `model` value in all API requests.

---

## Example Request

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello! This is Kokoro running locally.",
    "model": "kokoro-v1.0",
    "speaker_name": "af_heart",
    "speed": "normal"
  }' \
  --output speech.mp3
```

---

## Voices

Voices are identified by a two-character prefix that encodes language and gender:

| Prefix | Lang code (`/voices` key) | Example voices |
|---|---|---|
| `af_` / `am_` | `en-us` | `af_heart`, `af_bella`, `am_fenrir`, `am_michael` |
| `bf_` / `bm_` | `en-gb` | `bf_emma`, `bf_alice`, `bm_george`, `bm_daniel` |
| `jf_` / `jm_` | `ja` | `jf_alpha`, `jm_kumo` |
| `zf_` / `zm_` | `zh` | `zf_xiaobei`, `zm_yunxi` |
| `ef_` / `em_` | `es` | `ef_dora`, `em_alex` |
| `ff_` | `fr-fr` | `ff_siwis` |
| `hf_` / `hm_` | `hi` | `hf_alpha`, `hm_omega` |
| `if_` / `im_` | `it` | `if_sara`, `im_nicola` |
| `pf_` / `pm_` | `pt-br` | `pf_dora`, `pm_alex` |

See all voices via `GET /voices`.

### Full Voice List

**`en-us`** — American English
`af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**`en-gb`** — British English
`bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

**`ja`** — Japanese
`jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`

**`zh`** — Mandarin Chinese
`zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`

**`es`** — Spanish
`ef_dora`, `em_alex`, `em_santa`

**`fr-fr`** — French
`ff_siwis`

**`hi`** — Hindi
`hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`

**`it`** — Italian
`if_sara`, `im_nicola`

**`pt-br`** — Brazilian Portuguese
`pf_dora`, `pm_alex`, `pm_santa`

---

## Speed Control

| Value | Speed multiplier |
|---|---|
| `"slow"` | 0.8× |
| `"normal"` | 1.0× |
| `"fast"` | 1.3× |

---

## Limitations

- No voice cloning
- No voice design
- `temperature` and `seed` fields are accepted but have no effect (ONNX is deterministic)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Kokoro model files not found` | Download both `.onnx` and `.bin` files to `models/kokoro/` |
| `Unknown Kokoro voice` | Check `GET /voices` for valid voice names |
| Audio sounds unnatural | Try `af_heart` (blend of bella + sarah) for best quality |
