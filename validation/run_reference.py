"""Reference run: the default settings at dimension one, on CPU.

Run from the repository root. All artifacts go to --output, outside the source tree,
together with a results.json report of the resolved configurations and metrics.
"""
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from settings import make_config
from nnpd import execute, load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = make_config("default")
    cfg["problem"]["dimension"] = 1
    cfg["output"] = str(args.output.resolve())
    cfg["runtime"]["device"] = "cpu"
    start = time.perf_counter()
    paths = execute(cfg, load_experiment(cfg["application"]), settings_file=ROOT / "settings.py")
    report = {"profile": "default", "dimension": 1, "seconds": time.perf_counter() - start, "runs": []}
    for path in paths:
        record = json.loads((path / "run.json").read_text())
        report["runs"].append({"run": record, "metrics": json.loads((path / "metrics.json").read_text())})
    destination = args.output.resolve() / "results.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {destination}; {len(paths)} runs in {report['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
