"""Validation only: full legacy 1D settings and the mixed-parameter example.

Run from the repository root. Large runtime artifacts go outside the distributable
package. Reports and the exact resolved configurations are kept in validation/.
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
    parser.add_argument("profile", choices=["legacy_1d", "mixed"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = make_config("legacy" if args.profile == "legacy_1d" else "mixed")
    if args.profile == "legacy_1d":
        cfg["problem"]["dimension"] = 1
    cfg["output"] = str(args.output.resolve())
    cfg["runtime"]["device"] = "cpu"
    start = time.perf_counter()
    paths = execute(cfg, load_experiment(cfg["application"]), settings_file=ROOT / "settings.py")
    report = {"profile": args.profile, "seconds": time.perf_counter() - start, "runs": []}
    for path in paths:
        record = json.loads((path / "run.json").read_text())
        report["runs"].append({"run": record, "metrics": json.loads((path / "metrics.json").read_text())})
    destination = ROOT / "validation" / f"{args.profile}_results.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved {destination}; {len(paths)} runs in {report['seconds']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
