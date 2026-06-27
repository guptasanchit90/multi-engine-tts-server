import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    _MLX_AVAILABLE = True
except ImportError:
    pass

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install -r requirements.txt") from exc

from src.cache import ModelCache
from src.utils import (
    VOICES_DIR,
    SAMPLE_RATE,
    get_audio_duration,
    model_path as _model_path,
    resolve_voice,
    scan_wav_voices,
)

from .base import BaseEngine, register

MODELS_DIR = os.path.join(os.getcwd(), "models", "chatterbox")
MIN_REF_DURATION = 5.0

_MODELS: dict[str, str] = {
    "chatterbox-turbo-fp16": "Chatterbox-Turbo-TTS-fp16",
}

_model_cache = ModelCache(ttl=10, tag="chatterbox")


@register
class ChatterboxEngine(BaseEngine):
    @property
    def engine_name(self) -> str:
        return "chatterbox"

    def claims(self, model: str) -> bool:
        return model in _MODELS

    def list_models(self) -> list[dict]:
        folder = _MODELS.get("chatterbox-turbo-fp16", "")
        return [
            {
                "engine": "chatterbox",
                "model": "chatterbox-turbo-fp16",
                "mode": "clone",
                "available": _model_path(MODELS_DIR, folder) is not None,
            }
        ]

    def list_voices(self) -> dict:
        cloneable = scan_wav_voices(VOICES_DIR)
        return {"cloneable": cloneable} if cloneable else {}

    def validate(self, request: dict) -> None:
        voice_file = request.get("sample_voice_file")
        if not voice_file:
            raise HTTPException(
                status_code=422,
                detail="'sample_voice_file' is required for Chatterbox (voice cloning engine). "
                       "Specify a .wav file from the voices/ directory.",
            )
        resolved = resolve_voice(voice_file)
        if not resolved:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Voice file '{voice_file}' not found in voices/. "
                    "Place a .wav file there and retry."
                ),
            )
        duration = get_audio_duration(resolved)
        if duration < MIN_REF_DURATION:
            raise HTTPException(
                status_code=422,
                detail=f"Reference audio must be at least {MIN_REF_DURATION:.0f}s (got {duration:.1f}s)",
            )

    def generate(self, request: dict, tmp_dir: str) -> str:
        if not _MLX_AVAILABLE:
            raise HTTPException(
                status_code=422,
                detail="Chatterbox Turbo requires MLX (Apple Silicon only). Use the Kokoro or Piper engine instead.",
            )

        model_name = request["model"]
        temperature = request["temperature"]
        text = request["text"]

        folder = _MODELS.get(model_name)
        if not folder:
            raise HTTPException(status_code=422, detail=f"Unknown model '{model_name}'")

        resolved_path = _model_path(MODELS_DIR, folder)
        if not resolved_path:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model folder '{folder}' not found in {MODELS_DIR}. "
                    "Run: hf download mlx-community/Chatterbox-Turbo-TTS-fp16 --local-dir models/chatterbox/Chatterbox-Turbo-TTS-fp16"
                ),
            )

        mx.random.seed(request["effective_seed"])

        try:
            model = _model_cache.get_or_load(
                model_name, lambda: load_model(Path(resolved_path))
            )
        except HTTPException:
            raise
        except Exception as e:
            print(f"[chatterbox] Model load failed for '{model_name}': {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Model load failed: {e}. Ensure the model is downloaded and MLX is working.",
            )

        ref_audio = None
        voice_file = request.get("sample_voice_file")
        if voice_file:
            ref_audio = resolve_voice(voice_file)

        try:
            audio_chunks = []
            for result in model.generate(
                text=text,
                ref_audio=ref_audio,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.2,
                max_tokens=800,
                norm_loudness=True,
            ):
                audio_chunks.append(np.array(result.audio))

            if not audio_chunks:
                raise HTTPException(status_code=500, detail="TTS produced no audio output")

            audio = np.concatenate(audio_chunks)
            wav_path = os.path.join(tmp_dir, "audio_000.wav")
            sf.write(wav_path, audio, SAMPLE_RATE)

        except HTTPException:
            raise
        except Exception as e:
            print(f"[chatterbox] Generation failed for '{model_name}': {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

        if not os.path.exists(wav_path):
            raise HTTPException(status_code=500, detail="TTS produced no output file")

        return wav_path
