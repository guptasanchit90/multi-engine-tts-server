from typing import Protocol


class TTSEngine(Protocol):
    """
    Interface every TTS engine must satisfy.

    Adding a new engine:
      1. Create src/engines/<name>.py and implement this protocol.
      2. Register an instance in server.py's ENGINES dict.

    The engine is responsible for:
      - Knowing which model identifiers it owns (claims()).
      - Validating engine-specific fields in the request before generation.
      - Running inference and returning a path to a WAV file inside tmp_dir.
      - Cleaning up any resources it allocates (models, scratch files, etc.).

    The server handles everything else: temp dir lifecycle, WAV→MP3 conversion,
    response streaming, seed echoing.
    """

    def claims(self, model: str) -> bool:
        """Return True if this engine handles the given model identifier."""
        ...

    def list_models(self) -> list[dict]:
        """
        Return metadata for all models this engine knows about.
        Each dict must contain at minimum:
          { "engine": str, "model": str, "mode": str, "available": bool }
        """
        ...

    def list_voices(self) -> dict:
        """
        Return all voices this engine exposes, structured by category.

        Must return a dict of { category_label: [voice_id, ...] }.
        Categories are engine-defined — examples:
          - Built-in speaker groups: "American English", "British English", ...
          - Cloneable WAV files: "Cloneable voices"

        Return {} if the engine has no voices.
        """
        ...

    def validate(self, request: dict) -> None:
        """
        Validate engine-specific fields in the decoded request dict.
        Raise fastapi.HTTPException(status_code=422, ...) on any error.
        Called before generate() so heavy model loading is never reached on
        bad input.
        """
        ...

    def generate(self, request: dict, tmp_dir: str) -> str:
        """
        Run inference and write the output audio as `audio_000.wav` inside
        tmp_dir.  Return the full path to that WAV file.

        The request dict contains all fields from TTSRequest (text, model,
        speaker_name, voice_description, sample_voice_file, speed, seed,
        temperature) already validated by the server's Pydantic model and by
        validate().

        Raise fastapi.HTTPException on any runtime error.
        """
        ...
