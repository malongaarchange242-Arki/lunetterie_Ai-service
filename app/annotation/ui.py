import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Dict, Optional

from PIL import Image, ImageTk

from app.annotation.csv_manager import CSVManager
from app.annotation.dataset_manager import copy_to_class_folders
from app.annotation.labels import COULEURS, FORMES, GENRES, MATERIAUX, TYPE_MONTURE
from app.annotation.progress import ProgressManager

SECTION_DEFINITIONS = [
    ("forme", FORMES, ["1", "2", "3", "4", "5", "6", "7", "8"]),
    ("couleur", COULEURS, ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]),
    ("materiau", MATERIAUX, ["Q", "W", "E", "R", "T", "Y", "U"]),
    ("type", TYPE_MONTURE, ["Z", "X", "C", "V", "B"]),
    ("genre", GENRES, ["J", "K", "L", "M"]),
]


class AnnotationApp:
    def __init__(self, root: tk.Tk, image_dir: Path, csv_path: Path, progress_path: Path | None = None) -> None:
        self.root = root
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.progress_path = progress_path or image_dir.parent / "progress.json"
        self.csv_manager = CSVManager(csv_path)
        self.progress_manager = ProgressManager(self.progress_path)
        self.images = self._collect_images()
        self.index = self._resume_index()
        self.zoom = 1.0
        self.current_image: Optional[Path] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.selected_values: Dict[str, Optional[str]] = {field: None for field, _, _ in SECTION_DEFINITIONS}

        self.root.title("Annotation de montures")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")

        self._build_ui()
        self._bind_shortcuts()
        self._load_current_image()

    def _collect_images(self) -> list[Path]:
        if not self.image_dir.exists():
            return []
        return sorted([
            p for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

    def _resume_index(self) -> int:
        last_index = self.progress_manager.load()
        if last_index is None:
            return 0
        return max(0, min(last_index, len(self.images) - 1 if self.images else 0))

    def _build_ui(self) -> None:
        self.main_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.top_bar = tk.Frame(self.main_frame, bg="#2b2b2b")
        self.top_bar.pack(fill=tk.X, padx=10, pady=(10, 5))

        self.file_label = tk.Label(self.top_bar, text="Fichier", fg="white", bg="#2b2b2b")
        self.file_label.pack(side=tk.LEFT, padx=10, pady=8)

        self.progress_var = tk.StringVar(value="0/0")
        self.progress_label = tk.Label(self.top_bar, textvariable=self.progress_var, fg="white", bg="#2b2b2b")
        self.progress_label.pack(side=tk.RIGHT, padx=10, pady=8)

        self.image_canvas = tk.Canvas(self.main_frame, bg="#1e1e1e", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.image_canvas.bind("<Double-Button-1>", self._reset_zoom)
        self.image_canvas.bind("<MouseWheel>", self._zoom)

        self.controls_frame = tk.Frame(self.main_frame, bg="#2b2b2b")
        self.controls_frame.pack(fill=tk.X, padx=10, pady=(5, 10))

        self._build_attribute_buttons()

        self.status_var = tk.StringVar(value="Sélectionnez les attributs pour l'image courante")
        self.status_label = tk.Label(self.controls_frame, textvariable=self.status_var, fg="white", bg="#2b2b2b")
        self.status_label.pack(side=tk.LEFT, padx=10, pady=10)

        save_btn = tk.Button(self.controls_frame, text="Entrée - Enregistrer", width=22, command=self._save_and_next, bg="#1f6feb", fg="white")
        save_btn.pack(side=tk.RIGHT, padx=10, pady=10)

    def _build_attribute_buttons(self) -> None:
        self.attribute_frames = []

        for field_name, options, keys in SECTION_DEFINITIONS:
            frame = tk.LabelFrame(self.controls_frame, text=self._label_title(field_name), bg="#2b2b2b", fg="white", padx=10, pady=8)
            frame.pack(side=tk.LEFT, padx=8, pady=5)
            self.attribute_frames.append(frame)
            for value, key in zip(options, keys):
                btn = tk.Button(frames:=frame, text=f"{key} - {value}", width=18, command=lambda value=value, field=field_name: self._set_value(field, value))
                btn.pack(anchor=tk.W, pady=2)

    def _label_title(self, field_name: str) -> str:
        titles = {
            "forme": "Forme",
            "couleur": "Couleur",
            "materiau": "Matériau",
            "type": "Type",
            "genre": "Genre",
        }
        return titles.get(field_name, field_name)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Escape>", lambda event: self.root.destroy())
        self.root.bind("<Left>", lambda event: self._previous_image())
        self.root.bind("<Right>", lambda event: self._next_image())
        self.root.bind("s", lambda event: self._skip_image())
        self.root.bind("<Return>", lambda event: self._save_and_next())

        for field_name, options, keys in SECTION_DEFINITIONS:
            for idx, key in enumerate(keys):
                if idx >= len(options):
                    continue
                self.root.bind(key, lambda event, field=field_name, value=options[idx]: self._set_value(field, value))

    def _load_current_image(self) -> None:
        if not self.images:
            messagebox.showinfo("Info", "Aucune image à annoter.")
            self.root.destroy()
            return

        if self.index >= len(self.images):
            self.index = len(self.images) - 1

        self.current_image = self.images[self.index]
        self._display_image()
        self._refresh_progress()

    def _display_image(self) -> None:
        if self.current_image is None:
            return
        image = Image.open(self.current_image).convert("RGB")
        width, height = image.size
        resized = image.resize((int(width * self.zoom), int(height * self.zoom)))
        self.current_photo = ImageTk.PhotoImage(resized)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.current_photo)
        self.file_label.config(text=f"Fichier : {self.current_image.name}")

    def _zoom(self, event: tk.Event) -> None:
        if event.delta > 0:
            self.zoom = min(4.0, self.zoom + 0.1)
        else:
            self.zoom = max(0.5, self.zoom - 0.1)
        self._display_image()

    def _reset_zoom(self, event: tk.Event) -> None:
        self.zoom = 1.0
        self._display_image()

    def _refresh_progress(self) -> None:
        self.progress_var.set(f"{self.index + 1}/{len(self.images)}")

    def _set_value(self, field_name: str, value: str) -> None:
        self.selected_values[field_name] = value
        summary = ", ".join(f"{k}={v}" for k, v in self.selected_values.items() if v)
        self.status_var.set(summary or "Sélectionnez les attributs pour l'image courante")

    def _save_and_next(self) -> None:
        if not self.current_image:
            return
        if any(self.selected_values.get(field) is None for field in self.selected_values):
            messagebox.showerror("Erreur", "Veuillez sélectionner tous les attributs avant d'enregistrer.")
            return

        self._save_annotation(self.current_image.name, self.selected_values)
        self.progress_manager.save(self.index)
        self._next_image()

    def _save_annotation(self, image_name: str, labels: Dict[str, Optional[str]]) -> None:
        cleaned_labels = {key: value for key, value in labels.items() if value is not None}
        self.csv_manager.save_annotation(image_name, cleaned_labels)
        copy_to_class_folders(self.current_image, cleaned_labels)
        self.root.after(0, lambda: messagebox.showinfo("Succès", f"Annotation enregistrée pour {image_name}"))

    def _next_image(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
            self._reset_selection()
            self._load_current_image()
        else:
            messagebox.showinfo("Info", "Toutes les images ont été annotées.")

    def _previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._reset_selection()
            self._load_current_image()

    def _skip_image(self) -> None:
        self._reset_selection()
        self._next_image()

    def _reset_selection(self) -> None:
        self.selected_values = {field: None for field, _, _ in SECTION_DEFINITIONS}
        self.status_var.set("Sélectionnez les attributs pour l'image courante")


def main() -> None:
    root = tk.Tk()
    app = AnnotationApp(
        root,
        Path(__file__).resolve().parents[2] / "classification_dataset" / "crops",
        Path(__file__).resolve().parents[2] / "classification_dataset" / "annotations.csv",
        Path(__file__).resolve().parents[2] / "classification_dataset" / "progress.json",
    )
    root.mainloop()


if __name__ == "__main__":
    main()
