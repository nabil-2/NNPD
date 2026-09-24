"""Real torchrun worker used by integration tests, not a mocked distributed backend."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nnpd import execute, load_settings
from nnpd.core.storage import write_json

if __name__ == "__main__":
    folder = Path(sys.argv[1])
    paths = execute("run", settings_dir=folder)
    output = Path(load_settings(folder / "settings_training.py")["output"])
    write_json(output / f"worker-{os.environ.get('RANK', '0')}.json", {"completed": [str(path) for path in paths]})
