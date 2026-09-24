"""The command line and notebooks call the same public functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.settings import at
from .core.runner import STAGES, _load, check_feasible, execute, plan, verify_store


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and analyze from settings_training.py and "
                                                 "settings_analysis.py; the default action only prints a plan.")
    parser.add_argument("action", choices=("plan", *STAGES, "verify"), nargs="?", default="plan")
    parser.add_argument("--profile", help="Profile of both settings files (default: DEFAULT_PROFILE)")
    parser.add_argument("--settings-dir", type=Path, default=Path("."),
                        help="Folder with settings_training.py and settings_analysis.py (default: current folder)")
    args = parser.parse_args(argv)
    if args.action == "plan":
        jobs = plan(args.profile, args.settings_dir)
        print(json.dumps({"jobs": [{"id": job.id, "changed": job.changed,
                                    "changed_value": at(job.settings, job.changed) if job.changed else None,
                                    "member": job.member, "workload": job.workload} for job in jobs],
                          "total_models": len(jobs),
                          "sweep_rule": "baseline plus one changed knob; fixed cohort per configuration"}, indent=2))
        try:
            check_feasible(jobs)
        except ValueError as error:
            raise SystemExit(str(error)) from None
    elif args.action == "verify":
        root = _load(args.settings_dir, args.profile, with_analysis=False).output
        print(f"{root}: verified {verify_store(root)} artifacts")
    else:
        execute(args.action, profile=args.profile, settings_dir=args.settings_dir)


if __name__ == "__main__":
    main()
