from typing import Protocol

_STT_REGISTRY: list[type] = []


def register(cls):
    _STT_REGISTRY.append(cls)
    return cls


def discover() -> list:
    return [cls() for cls in _STT_REGISTRY]


class BaseSTTEngine:
    @property
    def engine_name(self) -> str:
        name = type(self).__name__
        if name.endswith("Engine"):
            name = name[:-6]
        return name.lower()


class STTEngine(Protocol):
    def claims(self, model: str) -> bool: ...

    def list_models(self) -> list[dict]: ...

    def validate(self, request: dict) -> None: ...

    def transcribe(
        self, audio_path: str, model: str, language: str | None, temperature: float
    ) -> dict: ...
