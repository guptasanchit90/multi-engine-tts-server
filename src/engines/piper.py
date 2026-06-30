import os
import wave
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from piper import PiperVoice
    from piper.config import SynthesisConfig
except ImportError as exc:
    raise ImportError("piper-tts is not installed. Run: pip install piper-tts") from exc

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

from .base import BaseEngine, register

MODELS_DIR = os.path.join(os.getcwd(), "models", "piper")


def _scan_voices() -> dict[str, str]:
    voices: dict[str, str] = {}
    if not os.path.exists(MODELS_DIR):
        return voices
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".onnx"):
            continue
        onnx_path = os.path.join(MODELS_DIR, fname)
        json_path = onnx_path + ".json"
        if os.path.exists(json_path):
            voice_name = fname[:-5]
            voices[voice_name] = onnx_path
    return voices


def _is_piper_model(model: str) -> bool:
    parts = model.split("-")
    if len(parts) >= 3 and "_" in parts[0]:
        return True
    return model in _scan_voices()


@register
class PiperEngine(BaseEngine):
    @property
    def engine_name(self) -> str:
        return "piper"

    def claims(self, model: str) -> bool:
        return _is_piper_model(model)

    def list_models(self) -> list[dict]:
        voices = _scan_voices()
        return [
            {
                "id": "piper",
                "name": "Piper TTS",
                "engine": "piper",
                "model": "",
                "mode": "speaker",
                "capabilities": ["speaker"],
                "description": "Lightweight ONNX-based TTS, 100+ downloadable voices",
                "size": "50-200 MB",
                "available": bool(voices),
                "voices": {"built_in": sorted(voices), "cloneable": []},
                "languages": sorted({v.split("-")[0].split("_")[0].lower() for v in voices}),
                "install": {
                    "source": "piper",
                    "commands": [
                        "python -m piper.download_voices --download-dir models/piper <voice_name>",
                    ],
                    "note": "Replace <voice_name> with e.g. en_US-lessac-medium",
                },
            }
        ]

    def list_voices(self) -> dict:
        voices = _scan_voices()
        grouped: dict[str, list[str]] = {}
        for name in sorted(voices):
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
        model = request["model"]
        text = request["text"]
        speed_val = request.get("speed_value", 1.0)
        length_scale = 1.0 / max(speed_val, 0.25)

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
            print(f"[piper] Failed to load voice '{model}': {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to load Piper voice '{model}': {e}"
            )

        wav_path = os.path.join(tmp_dir, "audio_000.wav")
        try:
            syn_config = SynthesisConfig(length_scale=length_scale)
            with wave.open(wav_path, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        except Exception as e:
            print(f"[piper] Synthesis failed for '{model}': {e}")
            raise HTTPException(status_code=500, detail=f"Piper synthesis failed: {e}")

        return wav_path
