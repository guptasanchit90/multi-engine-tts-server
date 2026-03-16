import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import soundfile as sf

try:
    from kokoro_onnx import Kokoro
except ImportError as exc:
    raise ImportError(
        "kokoro_onnx is not installed. Run: pip install kokoro-onnx"
    ) from exc

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.getcwd(), "models", "kokoro")

# Model files expected inside MODELS_DIR
ONNX_FILE   = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"

# Voice name → lang code passed to kokoro-onnx
# First letter of the voice name encodes language:
#   a = American English,  b = British English,  j = Japanese
#   z = Mandarin Chinese,  e = Spanish,           f = French
#   h = Hindi,             i = Italian,            p = Brazilian Portuguese
_LANG_MAP: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "zh",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

# All v1.0 voices from VOICES.md, keyed by the lang code from _LANG_MAP
_VOICES: dict[str, list[str]] = {
    "en-us": [
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "en-gb": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "ja":    ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo"],
    "zh":    ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
              "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"],
    "es":    ["ef_dora", "em_alex", "em_santa"],
    "fr-fr": ["ff_siwis"],
    "hi":    ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
    "it":    ["if_sara", "im_nicola"],
    "pt-br": ["pf_dora", "pm_alex", "pm_santa"],
}

_ALL_VOICES: set[str] = {v for vs in _VOICES.values() for v in vs}

# The single model identifier the server uses to route to this engine
MODEL_ID = "kokoro-v1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _onnx_path() -> str:
    return os.path.join(MODELS_DIR, ONNX_FILE)


def _voices_path() -> str:
    return os.path.join(MODELS_DIR, VOICES_FILE)


def _model_available() -> bool:
    return os.path.exists(_onnx_path()) and os.path.exists(_voices_path())


def _parse_voices(voice_input: str) -> tuple[list[str], list[float] | None]:
    """
    Parse voice input that can be:
      - Single voice: "af_heart"
      - Multiple voices: "af_heart,am_michael"
      - Weighted voices: "af_heart:0.6,am_michael:0.4"
    
    Returns (voices, weights) where weights is None for equal blend.
    """
    parts = [p.strip() for p in voice_input.split(",")]
    voices = []
    weights = []
    
    for part in parts:
        if ":" in part:
            voice, weight = part.rsplit(":", 1)
            voices.append(voice.strip())
            try:
                weights.append(float(weight.strip()) / 100)
            except ValueError:
                weights.append(float(weight.strip()))
        else:
            voices.append(part)
    
    if weights and len(weights) == len(voices):
        return voices, weights
    return voices, None


def _all_same_lang(voices: list[str]) -> bool:
    """Check if all voices share the same language prefix."""
    if not voices:
        return True
    prefixes = {v[0] for v in voices}
    return len(prefixes) == 1


def _lang_for_voice(voice: str) -> str:
    """Derive the language code from the voice name prefix."""
    prefix = voice[0] if voice else "a"
    return _LANG_MAP.get(prefix, "en-us")


def _add_pauses(text: str, sample_rate: int) -> np.ndarray:
    """
    Insert silence into audio based on punctuation in text.
    
    Returns silence array to append between segments.
    """
    PAUSE_MAP = {
        ".": 0.5,
        "...": 1.2,
        "?": 0.5,
        "!": 0.6,
        ",": 0.2,
        ";": 0.3,
    }
    newline_pause = 0.4

    total_silence = 0.0
    i = 0
    while i < len(text):
        matched = False
        for punct in sorted(PAUSE_MAP.keys(), key=len, reverse=True):
            if text[i:].startswith(punct):
                total_silence += PAUSE_MAP[punct]
                i += len(punct)
                matched = True
                break
        if not matched:
            if text[i] == "\n":
                total_silence += newline_pause
            i += 1

    if total_silence <= 0:
        return np.array([], dtype=np.float32)

    return np.zeros(int(sample_rate * total_silence), dtype=np.float32)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class KokoroEngine:
    """
    TTS engine wrapping Kokoro-82M via kokoro-onnx (ONNX Runtime).

    Model files required in models/kokoro/:
      - kokoro-v1.0.onnx   (~300 MB)
      - voices-v1.0.bin

    Download from:
      https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0

    Kokoro supports custom voices, speed control, and 9 languages.
    It does NOT support voice cloning or voice design.
    """

    # --- TTSEngine protocol ---

    def claims(self, model: str) -> bool:
        return model == MODEL_ID

    def list_models(self) -> list[dict]:
        return [
            {
                "engine":    "kokoro",
                "model":     MODEL_ID,
                "mode":      "speaker",
                "available": _model_available(),
            }
        ]

    def list_voices(self) -> dict:
        """Return all built-in Kokoro voices grouped by BCP-47 language code."""
        return {lang: sorted(voices) for lang, voices in _VOICES.items()}

    def validate(self, request: dict) -> None:
        if not _model_available():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Kokoro model files not found in {MODELS_DIR}. "
                    f"Download {ONNX_FILE} and {VOICES_FILE} from "
                    "https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0"
                ),
            )

        speaker = request.get("speaker_name") or "af_heart"
        voices, weights = _parse_voices(speaker)

        for v in voices:
            if v not in _ALL_VOICES:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown Kokoro voice '{v}'. See GET /voices for the full list.",
                )

        if len(voices) > 1 and not _all_same_lang(voices):
            raise HTTPException(
                status_code=422,
                detail="Cannot blend voices from different languages.",
            )

    def generate(self, request: dict, tmp_dir: str) -> str:
        """Run inference and write audio_000.wav into tmp_dir."""
        voice  = request.get("speaker_name") or "af_heart"
        speed  = request["speed_value"]   # float, resolved by server
        text   = request["text"]
        add_pauses = request.get("add_pauses", True)

        voices, weights = _parse_voices(voice)
        lang = _lang_for_voice(voices[0])

        try:
            kokoro = Kokoro(_onnx_path(), _voices_path())

            if len(voices) == 1:
                voice_param = voices[0]
            else:
                blended = None
                for i, v in enumerate(voices):
                    style = kokoro.get_voice_style(v)
                    w = weights[i] if weights else (1.0 / len(voices))
                    if blended is None:
                        blended = style * w
                    else:
                        blended = np.add(blended, style * w)
                voice_param = blended
            assert voice_param is not None

            samples, sample_rate = kokoro.create(
                text,
                voice=voice_param,
                speed=speed,
                lang=lang,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Kokoro generation failed: {e}")

        samples = np.array(samples)
        if add_pauses:
            pause_samples = _add_pauses(text, sample_rate)
            if len(pause_samples) > 0:
                samples = np.concatenate([samples, pause_samples])

        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        try:
            sf.write(wav_path, samples, sample_rate)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write WAV: {e}")

        return wav_path
