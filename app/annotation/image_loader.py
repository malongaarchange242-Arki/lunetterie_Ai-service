from pathlib import Path
from typing import List


class ImageLoader:
    def __init__(self, image_dir: Path) -> None:
        self.image_dir = image_dir

    def list_images(self) -> List[Path]:
        if not self.image_dir.exists():
            return []
        return sorted(
            [p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
