"""The command line and notebooks call the same public functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.config import load_settings, at
from .core.runner import execute, plan, verify_store
from .core.api import load_experiment


def main(argv=None):
    parser = argparse.ArgumentParser(description="Modular experiment runner; default action only prints a plan.")
    parser.add_argument("action", choices=("plan", "run", "train", "analyze", "verify"), nargs="?", default="plan")
    parser.add_argument("--config", type=Path, default=Path("settings.py"), help="Trusted Python settings file")
    parser.add_argument("--profile", help="Profile from settings.py")
    parser.add_argument("--application", help="Override application import path module:class from settings.py")
    args = parser.parse_args(argv)
    config = load_settings(args.config, args.profile)
    if args.application:
        config["application"] = args.application
    experiment = load_experiment(config["application"])
    if args.action == "plan":
        jobs = plan(config, experiment)
        print(json.dumps({"jobs": [{"id": job.id, "changed": job.changed,
                                    "changed_value": at(job.config, job.changed) if job.changed else None, "member": job.member,
                                    "workload": job.workload} for job in jobs],
                          "total_models": len(jobs),
                          "sweep_rule": "baseline plus one changed knob; fixed cohort per configuration"}, indent=2))
    elif args.action == "verify":
        roots = sorted({job.config["output"] for job in plan(config, experiment)})
        for root in roots:
            print(f"{root}: verified {verify_store(root)} artifacts")
    else:
        execute(config, experiment, stage=args.action, settings_file=args.config)


if __name__ == "__main__":
    main()
