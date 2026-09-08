"""Real torchrun worker used by integration tests, not a mocked distributed backend."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nnpd import execute
from nnpd.core.storage import write_json
from gaussian import GaussianExperiment

if __name__ == "__main__":
    config = json.loads(Path(sys.argv[1]).read_text())
    paths = execute(config, GaussianExperiment())
    write_json(Path(config["output"]) / f"worker-{os.environ.get('RANK', '0')}.json",
               {"completed": [str(path) for path in paths]})
