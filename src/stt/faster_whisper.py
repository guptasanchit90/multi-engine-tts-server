import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def _faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False

try:
    from fastapi import HTTPException
except ImportError as exc:
    raise ImportError("fastapi is not installed. Run: pip install fastapi") from exc

from .base import BaseSTTEngine, register

MODELS_DIR = os.path.join(os.getcwd(), "models", "faster-whisper")

_REPOS: dict[str, str] = {
    "faster-whisper-tiny": "Systran/faster-whisper-tiny",
    "faster-whisper-base": "Systran/faster-whisper-base",
    "faster-whisper-small": "Systran/faster-whisper-small",
    "faster-whisper-medium": "Systran/faster-whisper-medium",
    "faster-whisper-large-v3": "Systran/faster-whisper-large-v3",
}

_MODEL_SIZES: dict[str, str] = {
    "faster-whisper-tiny": "~75 MB",
    "faster-whisper-base": "~150 MB",
    "faster-whisper-small": "~500 MB",
    "faster-whisper-medium": "~1.5 GB",
    "faster-whisper-large-v3": "~3 GB",
}


def _model_local_path(model_id: str) -> str | None:
    for name in (model_id.removeprefix("faster-whisper-"), model_id):
        path = os.path.join(MODELS_DIR, name)
        if os.path.isdir(path):
            return path
    return None


def _resolve_model_path(model_id: str) -> str | None:
    return _model_local_path(model_id) or _REPOS.get(model_id)


@register
class FasterWhisperEngine(BaseSTTEngine):
    @property
    def engine_name(self) -> str:
        return "faster_whisper"

    def claims(self, model: str) -> bool:
        return model in _REPOS

    def list_models(self) -> list[dict]:
        return [
            {
                "id": mid,
                "name": f"Faster Whisper {mid.removeprefix('faster-whisper-').replace('-', ' ').title()}",
                "engine": self.engine_name,
                "model": mid,
                "mode": "stt",
                "capabilities": ["transcribe"],
                "description": f"OpenAI Whisper via CTranslate2 ({mid})",
                "available": _faster_whisper_available() and _model_local_path(mid) is not None,
                "mlx_required": False,
                "languages": ["en", "multi"],
                "size": _MODEL_SIZES.get(mid, ""),
                "install": {
                    "source": "HuggingFace",
                    "url": f"https://huggingface.co/{_REPOS[mid]}",
                    "commands": [
                        f"hf download {_REPOS[mid]} --local-dir models/faster-whisper/{mid.removeprefix('faster-whisper-')}",
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
        if not _faster_whisper_available():
            raise HTTPException(
                status_code=500,
                detail="faster-whisper is not installed. Run: pip install faster-whisper",
            )

    def transcribe(
        self, audio_path: str, model: str, language: str | None, temperature: float
    ) -> dict:
        model_path_or_repo = _resolve_model_path(model)
        if model_path_or_repo is None:
            raise HTTPException(
                status_code=422,
                detail=f"STT model '{model}' not found. Known repos: {_REPOS}",
            )

        from faster_whisper import WhisperModel

        whisper = WhisperModel(
            model_path_or_repo,
            device="auto",
            compute_type="auto",
        )

        segments, info = whisper.transcribe(
            audio_path,
            language=language,
            temperature=temperature,
            beam_size=5,
        )

        all_segments = list(segments)
        text = "".join(seg.text for seg in all_segments)
        detected_lang = info.language if info else (language or "")

        return {
            "text": text.strip(),
            "segments": [
                {
                    "id": i,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                }
                for i, seg in enumerate(all_segments)
            ],
            "detected_language": detected_lang,
        }
