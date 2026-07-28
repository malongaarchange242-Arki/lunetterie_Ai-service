import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.annotation.dataset_manager import create_structure
from app.annotation.ui import main as run_ui


if __name__ == "__main__":
    create_structure()
    run_ui()
