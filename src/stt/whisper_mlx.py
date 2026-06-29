import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_MLX_WHISPER_AVAILABLE = False
try:
    import mlx_whisper

    _MLX_WHISPER_AVAILABLE = True
except ImportError:
    pass

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

from src.cache import ModelCache

from .base import BaseSTTEngine, register

MODELS_DIR = os.path.join(os.getcwd(), "models", "whisper")

_REPOS: dict[str, str] = {
    "whisper-tiny": "mlx-community/whisper-tiny",
    "whisper-base": "mlx-community/whisper-base-mlx",
    "whisper-small": "mlx-community/whisper-small-mlx",
    "whisper-medium": "mlx-community/whisper-medium-mlx",
    "whisper-large-v3": "mlx-community/whisper-large-v3-mlx",
}

_MODEL_SIZES: dict[str, str] = {
    "whisper-tiny": "~75 MB",
    "whisper-base": "~150 MB",
    "whisper-small": "~500 MB",
    "whisper-medium": "~1.5 GB",
    "whisper-large-v3": "~3 GB",
}


def _model_path(model_id: str) -> str | None:
    full = os.path.join(MODELS_DIR, model_id)
    return full if os.path.isdir(full) else None


_model_cache = ModelCache(ttl=15, tag="whisper_mlx")


@register
class WhisperMlxEngine(BaseSTTEngine):
    @property
    def engine_name(self) -> str:
        return "whisper_mlx"

    def claims(self, model: str) -> bool:
        return model in _REPOS

    def list_models(self) -> list[dict]:
        return [
            {
                "id": mid,
                "name": f"Whisper {mid.removeprefix('whisper-').replace('-', ' ').title()}",
                "engine": self.engine_name,
                "model": mid,
                "mode": "stt",
                "capabilities": ["transcribe"],
                "description": f"OpenAI Whisper via MLX ({mid})",
                "available": _MLX_WHISPER_AVAILABLE and _model_path(mid) is not None,
                "languages": ["en", "multi"],
                "size": _MODEL_SIZES.get(mid, ""),
                "install": {
                    "commands": [
                        f"hf download {_REPOS[mid]} --local-dir models/whisper/{mid}",
                    ],
                },
            }
            for mid in _REPOS
        ]

    def list_voices(self) -> dict:
        return {}

    def validate(self, request: dict) -> None:
        if not request.get("model"):
            raise HTTPException(status_code=422, detail="'model' is required")
        if request["model"] not in _REPOS:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown STT model '{request['model']}'. Known: {sorted(_REPOS)}",
            )
        if not _MLX_WHISPER_AVAILABLE:
            raise HTTPException(
                status_code=500,
                detail="mlx-whisper is not installed. Run: pip install mlx-whisper",
            )
        mpath = _model_path(request["model"])
        if mpath is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Model '{request['model']}' not found in {MODELS_DIR}. "
                    f"Run: hf download {_REPOS[request['model']]} "
                    f"--local-dir models/whisper/{request['model']}"
                ),
            )

    def transcribe(
        self, audio_path: str, model: str, language: str | None, temperature: float
    ) -> dict:
        mpath = _model_path(model)
        if mpath is None:
            raise HTTPException(status_code=422, detail=f"STT model '{model}' not found locally")

        decode_opts = {}
        if language:
            decode_opts["language"] = language

        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=mpath,
            temperature=temperature,
            verbose=None,
            **decode_opts,
        )

        return {
            "text": result.get("text", "").strip(),
            "segments": result.get("segments", []),
            "detected_language": result.get("language", language or ""),
        }
