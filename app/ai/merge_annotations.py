from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def merge_annotations(shape_path: str = "classification_dataset/annotations/annotations_shape.csv", mount_path: str = "classification_dataset/annotations/annotations_mount.csv", output_path: str = "classification_dataset/annotations/annotations_complete.csv") -> list[dict[str, str]]:
    print("📊 Fusion des annotations...")

    with Path(shape_path).open("r", encoding="utf-8", newline="") as handle:
        shape_rows = list(csv.DictReader(handle))
    with Path(mount_path).open("r", encoding="utf-8", newline="") as handle:
        mount_rows = list(csv.DictReader(handle))

    print(f"  - Formes: {len(shape_rows)} images")
    print(f"  - Types: {len(mount_rows)} images")

    mount_by_filename = {row["filename"]: row.get("mount_type", "") for row in mount_rows}
    merged_records: list[dict[str, str]] = []
    for row in shape_rows:
        filename = row.get("filename", "")
        merged_records.append({
            "filename": filename,
            "shape": row.get("shape", ""),
            "mount_type": mount_by_filename.get(filename, ""),
        })

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "shape", "mount_type"])
        writer.writeheader()
        writer.writerows(merged_records)

    print(f"\n✅ Fichier fusionné: {output_file}")
    print(f"   {len(merged_records)} images annotées")
    print(f"\n📊 Distribution des formes:")
    print(Counter(record["shape"] for record in merged_records if record.get("shape")))
    print(f"\n📊 Distribution des types:")
    print(Counter(record["mount_type"] for record in merged_records if record.get("mount_type")))

    return merged_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="classification_dataset/annotations/annotations_shape.csv")
    parser.add_argument("--mount", type=str, default="classification_dataset/annotations/annotations_mount.csv")
    parser.add_argument("--output", type=str, default="classification_dataset/annotations/annotations_complete.csv")
    args = parser.parse_args()
    merge_annotations(args.shape, args.mount, args.output)


if __name__ == "__main__":
    main()
