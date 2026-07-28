import csv
from pathlib import Path
from typing import Dict, List


class CSVManager:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["image", "forme", "couleur", "materiau", "type", "genre"])

    def append_row(self, row: Dict[str, str]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "forme", "couleur", "materiau", "type", "genre"])
            writer.writerow(row)

    def save_annotation(self, image_name: str, labels: Dict[str, str]) -> None:
        row = {
            "image": image_name,
            "forme": labels.get("forme", ""),
            "couleur": labels.get("couleur", ""),
            "materiau": labels.get("materiau", ""),
            "type": labels.get("type", ""),
            "genre": labels.get("genre", ""),
        }
        self.append_row(row)

    def read_rows(self) -> List[Dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
