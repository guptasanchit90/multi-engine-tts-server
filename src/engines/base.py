from typing import Protocol

_REGISTRY: list[type] = []


def register(cls):
    _REGISTRY.append(cls)
    return cls


def discover() -> list:
    return [cls() for cls in _REGISTRY]


class BaseEngine:
    @property
    def engine_name(self) -> str:
        name = type(self).__name__
        if name.endswith("Engine"):
            name = name[:-6]
        return name.lower()

    def list_voices(self) -> dict:
        from src.utils import scan_wav_voices
        cloneable = scan_wav_voices()
        return {"cloneable": cloneable} if cloneable else {}


class TTSEngine(Protocol):
    def claims(self, model: str) -> bool: ...

    def list_models(self) -> list[dict]: ...

    def list_voices(self) -> dict: ...

    def validate(self, request: dict) -> None: ...

    def generate(self, request: dict, tmp_dir: str) -> str: ...
