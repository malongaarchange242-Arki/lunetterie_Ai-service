from pathlib import Path

from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    MODEL_PATH_DETECTION: str = str(BASE_DIR / "app/models/detection/best.pt")
    MODEL_PATH_CLASSIFICATION: str = str(BASE_DIR / "app/models/classification/best_shape_model_balanced.pth")
    MODEL_VERSION: str = "1.0.0"
    CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: str = "cpu"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"

    class Config:
        env_file = str(BASE_DIR / ".env")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.MODEL_PATH_DETECTION = self._resolve_path(self.MODEL_PATH_DETECTION)
        self.MODEL_PATH_CLASSIFICATION = self._resolve_path(self.MODEL_PATH_CLASSIFICATION)

    def _resolve_path(self, value: str) -> str:
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((BASE_DIR / path).resolve())


settings = Settings()
