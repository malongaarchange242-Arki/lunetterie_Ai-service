import json
from pathlib import Path
from typing import Optional


class ProgressManager:
    def __init__(self, progress_path: Path) -> None:
        self.progress_path = progress_path
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Optional[int]:
        if not self.progress_path.exists():
            return None
        with self.progress_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("last_index")

    def save(self, index: int) -> None:
        with self.progress_path.open("w", encoding="utf-8") as handle:
            json.dump({"last_index": index}, handle, indent=2)
