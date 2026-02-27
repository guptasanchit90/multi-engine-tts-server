import os
import wave
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
except ImportError as exc:
    raise ImportError(
        "piper-tts is not installed. Run: pip install piper-tts"
    ) from exc

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = os.path.join(os.getcwd(), "models", "piper")

# Speed → length_scale (Piper inverts the concept: longer = slower)
_LENGTH_SCALE: dict[str, float] = {
    "slow":   1.4,   # 1/0.8 — longer duration = slower speech
    "normal": 1.0,
    "fast":   0.75,  # 1/1.3 — shorter duration = faster speech
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scan_voices() -> dict[str, str]:
    """
    Scan MODELS_DIR for Piper voice files.
    Returns { voice_name: onnx_path } for every .onnx that has a matching .onnx.json.
    Read at call-time so newly downloaded voices appear without a restart.
    """
    voices: dict[str, str] = {}
    if not os.path.exists(MODELS_DIR):
        return voices
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".onnx"):
            continue
        onnx_path = os.path.join(MODELS_DIR, fname)
        json_path = onnx_path + ".json"
        if os.path.exists(json_path):
            voice_name = fname[:-5]  # strip .onnx
            voices[voice_name] = onnx_path
    return voices


def _is_piper_model(model: str) -> bool:
    """
    A model string belongs to Piper if it matches the Piper naming convention
    (<lang_locale>-<name>-<quality>) OR if a corresponding .onnx file is
    already present on disk.
    """
    # Naming convention: en_US-lessac-medium, de_DE-thorsten-low, etc.
    parts = model.split("-")
    if len(parts) >= 3 and "_" in parts[0]:
        return True
    # Fallback: voice file is present
    return model in _scan_voices()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PiperEngine:
    """
    TTS engine wrapping Piper (OHF-Voice/piper1-gpl) via piper-tts.

    Voice model files live in models/piper/ and are scanned at request time,
    so newly downloaded voices are available immediately without restarting.

    Each voice requires two files:
      - <voice_name>.onnx
      - <voice_name>.onnx.json

    Download voices with:
      python -m piper.download_voices --download-dir models/piper <voice_name>

    Example voice names:
      en_US-lessac-medium   (English US, medium quality)
      en_GB-alba-medium     (English UK)
      de_DE-thorsten-low    (German)
      fr_FR-siwis-medium    (French)

    Browse all voices: https://huggingface.co/rhasspy/piper-voices
    """

    # --- TTSEngine protocol ---

    def claims(self, model: str) -> bool:
        return _is_piper_model(model)

    def list_models(self) -> list[dict]:
        voices = _scan_voices()
        if not voices:
            return [
                {
                    "engine": "piper",
                    "model": "(no voices downloaded)",
                    "mode": "speaker",
                    "available": False,
                }
            ]
        return [
            {
                "engine": "piper",
                "model": name,
                "mode": "speaker",
                "available": True,
            }
            for name in sorted(voices)
        ]

    def list_voices(self) -> dict:
        """Return downloaded Piper voices grouped by language locale."""
        voices = _scan_voices()
        grouped: dict[str, list[str]] = {}
        for name in sorted(voices):
            # voice name: en_US-lessac-medium → locale en_US
            locale = name.split("-")[0] if "-" in name else "other"
            grouped.setdefault(locale, []).append(name)
        return grouped

    def validate(self, request: dict) -> None:
        model = request["model"]
        voices = _scan_voices()

        if model not in voices:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Piper voice '{model}' not found in {MODELS_DIR}. "
                    f"Download it with: python -m piper.download_voices --download-dir models/piper {model}"
                ),
            )

    def generate(self, request: dict, tmp_dir: str) -> str:
        """Run Piper inference; write audio_000.wav into tmp_dir."""
        model = request["model"]
        text = request["text"]
        speed_key = request["speed"]           # "slow" | "normal" | "fast"
        length_scale = _LENGTH_SCALE.get(speed_key, 1.0)

        voices = _scan_voices()
        onnx_path = voices.get(model)
        if not onnx_path:
            raise HTTPException(
                status_code=422,
                detail=f"Piper voice '{model}' disappeared from {MODELS_DIR}",
            )

        try:
            voice = PiperVoice.load(onnx_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load Piper voice '{model}': {e}")

        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        try:
            syn_config = SynthesisConfig(length_scale=length_scale)
            with wave.open(wav_path, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Piper synthesis failed: {e}")

        return wav_path
