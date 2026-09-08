"""Plan first; cache data and training; compute only requested analysis products."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import platform
import traceback
import zipfile

import numpy as np
import torch

from .api import Context, Experiment, implementation_hash
from .config import digest, expand_sweep
from .runtime import Runtime, seed_all
from .storage import Artifact, Store, file_hash, safe_name, utc_now, write_json


@dataclass(frozen=True)
class Job:
    config: dict
    member: dict
    changed: str | None
    workload: dict

    @property
    def id(self) -> str:
        return digest({"config": self.config, "member": self.member})

    def as_dict(self) -> dict:
        return {"id": self.id, "changed": self.changed, "member": self.member,
                "config": self.config, "workload": self.workload}


def _validate_hooks(experiment):
    products = experiment.products()
    if set(products) & {"training", "model"}:
        raise ValueError("training and model are reserved root datasets.")
    complete, active = {"training", "model"}, []

    def visit(name):
        safe_name(name)
        if name in complete:
            return
        if name in active:
            raise ValueError("Dataset dependency cycle: " + " -> ".join([*active, name]))
        if name not in products:
            raise KeyError(f"Unknown dataset dependency {name!r}.")
        active.append(name)
        for dependency in products[name].needs:
            visit(dependency)
        active.pop()
        complete.add(name)

    for name in products:
        visit(name)
    for hooks in (experiment.metrics(), experiment.plots()):
        for name, hook in hooks.items():
            safe_name(name)
            for dependency in hook.needs:
                visit(dependency)


def plan(config: dict, experiment: Experiment) -> list[Job]:
    """Validate *every* option before starting the first run."""
    _validate_hooks(experiment)
    jobs = []
    for variant in expand_sweep(config):
        cfg = variant.config
        experiment.validate(cfg)
        for category, registry in (("metrics", experiment.metrics()), ("plots", experiment.plots())):
            unknown = set(cfg[category]) - registry.keys()
            if unknown:
                raise ValueError(f"Unknown {category}: {sorted(unknown)}")
            if len(cfg[category]) != len(set(cfg[category])):
                raise ValueError(f"Duplicate names in {category}.")
        members = experiment.members(cfg)
        if not members:
            raise ValueError("An experiment must have at least one cohort member.")
        if len({digest(member) for member in members}) != len(members):
            raise ValueError("Duplicate cohort members.")
        for member in members:
            jobs.append(Job(cfg, member, variant.changed, experiment.estimate(cfg, member)))
    return jobs


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _source_snapshot(store: Store, experiment: Experiment, settings_file=None) -> str:
    """Archive our own implementation, not the uploaded legacy repository."""
    package = Path(__file__).resolve().parents[1]
    application = Path(inspect.getfile(type(experiment))).resolve().parent
    files = {f"nnpd/{path.relative_to(package)}": path for path in package.rglob("*.py")}
    for path in application.rglob("*.py"):
        files[f"{application.name}/{path.relative_to(application)}"] = path
    for name in ("settings.py", "run.py", "pyproject.toml"):
        path = application.parent / name
        if path.exists():
            files[name] = path
    if settings_file is not None:
        path = Path(settings_file).resolve()
        if path != application.parent / "settings.py":
            files[f"configuration/{path.name}"] = path
    key = digest({name: file_hash(path) for name, path in sorted(files.items())})
    folder = store.root / "source"
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{key}.zip"
    with store.lock("source", key):
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, path in sorted(files.items()):
                    archive.write(path, name)
            temporary.replace(destination)
    return key


def _prepare(context: Context, stage: str) -> None:
    experiment = context.experiment
    recipe = {"experiment": experiment.name, "stage": "training",
              "settings": experiment.signature("data", context),
              "implementation": implementation_hash(experiment.sources("data"), experiment.version)}
    key = digest(recipe)
    if stage == "analyze":
        artifact = context.store.existing("training", key)
        if artifact is None:
            raise FileNotFoundError("No matching training data. Run the 'train' or 'run' command first.")
    else:
        artifact = context.store.get("training", key, recipe,
                                     lambda writer: experiment.sample_training(context, writer))
    context.artifacts["training"] = artifact
    recipe = {"experiment": experiment.name, "stage": "model", "training": artifact.key,
              "settings": experiment.signature("model", context),
              "backend": context.runtime.training_signature,
              "implementation": implementation_hash(experiment.sources("model"), experiment.version)}
    key = digest(recipe)
    runtime = context.runtime
    lock = context.store.lock("model", key) if runtime.leader else nullcontext()
    with lock:
        cached = context.store.existing("model", key) if runtime.leader else None
        ready = runtime.broadcast(cached is not None)
        if not ready:
            if stage == "analyze":
                raise FileNotFoundError("No matching model checkpoint; analysis never trains implicitly.")
            seed_all(experiment.model_seed(context))
            model = experiment.build_model(context).to(runtime.device)
            history = experiment.train(context, model)
            if runtime.leader:
                with context.store.transaction("model", key, recipe) as writer:
                    writer.weights(model)
                    writer.json("history", history)
                    writer.finish("model", key, {"model_seed": experiment.model_seed(context)}, recipe)
                context._model = model.eval()
            runtime.barrier()
    artifact = context.store.existing("model", key)
    if artifact is None:
        raise RuntimeError("Model artifact was not committed.")
    context.artifacts["model"] = artifact


def analyze(context: Context) -> dict:
    """Evaluate selected hooks on an existing context; no training occurs here."""
    metrics = {}
    for name in context.config["metrics"]:
        hook = context.experiment.metrics()[name]
        value = hook.compute(context, context.dependencies(hook.needs))
        metrics[name] = _json_value(value)
        write_json(context.run_dir / "metrics" / f"{safe_name(name)}.json", metrics[name])
    if context.config["plots"]:
        from matplotlib import pyplot as plt
        for name in context.config["plots"]:
            hook = context.experiment.plots()[name]
            figures = hook.draw(context, context.dependencies(hook.needs))
            try:
                for stem, figure in figures.items():
                    path = context.run_dir / "plots" / safe_name(name) / f"{safe_name(stem)}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(path, dpi=context.config["plotting"]["dpi"], bbox_inches="tight")
            finally:
                for figure in figures.values():
                    plt.close(figure)
    write_json(context.run_dir / "metrics.json", metrics)
    save_references(context)
    return metrics


def save_references(context: Context) -> None:
    write_json(context.run_dir / "artifacts.json", {
        "store_relative": os.path.relpath(context.store.root, context.run_dir),
        "items": {name: os.path.relpath(artifact.path, context.run_dir)
                  for name, artifact in context.artifacts.items()},
    })


def execute(config: dict, experiment: Experiment, *, stage: str = "run", settings_file=None) -> list[Path]:
    """Same public entry point for scripts and notebooks.

    'jobs': torchrun workers process disjoint complete jobs, including inference.
    'ddp': workers train each model together; only rank zero runs analysis hooks.
    Checkpoints are stage-level: an interrupted, uncommitted training stage restarts.
    """
    if stage not in {"train", "run", "analyze"}:
        raise ValueError("stage must be train, run, or analyze.")
    jobs = plan(config, experiment)
    if stage != "train":
        excessive = [(job.id, job.workload["errors"]) for job in jobs if job.workload.get("errors")]
        if excessive:
            raise ValueError(f"Workload exceeds configured guards: {excessive}. Edit settings explicitly.")
    runtime_config = jobs[0].config["runtime"]
    if any(job.config["runtime"] != runtime_config for job in jobs):
        raise ValueError("Runtime settings cannot vary within one launch; use separate launches.")
    runtime = Runtime(runtime_config)
    completed = []
    try:
        for index, job in enumerate(jobs):
            if not runtime.ddp and index % runtime.world_size != runtime.rank:
                continue
            store = Store(job.config["output"])
            variant = (job.changed or "baseline").replace(".", "_")
            config_id = digest(job.config)[:12]
            member_name = safe_name(job.member.get("name", "member"))
            backend_id = digest(runtime.training_signature)[:8]
            run_dir = store.root / "runs" / f"{variant}-{config_id}" / f"{member_name}-{job.id[:8]}-{backend_id}"
            context = Context(experiment, job.config, job.member, store, runtime, run_dir)
            if runtime.leader:
                run_dir.mkdir(parents=True, exist_ok=True)
                source = _source_snapshot(store, experiment, settings_file)
                write_json(run_dir / "run.json", {**job.as_dict(), "experiment": experiment.name,
                           "source_archive": source, "python": platform.python_version(),
                           "backend": runtime.training_signature, "numpy": np.__version__,
                           "started": utc_now()})
                write_json(run_dir / "status.json", {"state": "running", "stage": stage, "time": utc_now()})
                print(f"[{index + 1}/{len(jobs)}] {variant}: {member_name}", flush=True)
            try:
                _prepare(context, stage)
                if runtime.leader:
                    save_references(context)
                    if stage != "train":
                        analyze(context)
                    write_json(run_dir / "status.json", {
                        "state": "trained" if stage == "train" else "complete", "time": utc_now()})
                    completed.append(run_dir)
                runtime.barrier()
            except Exception:
                if runtime.leader:
                    write_json(run_dir / "status.json", {
                        "state": "failed", "time": utc_now(), "traceback": traceback.format_exc()})
                raise
    finally:
        runtime.close()
    return completed


def restore(run_dir: str | Path, experiment: Experiment, *, device: str = "cpu") -> Context:
    """Open a saved run for arbitrary Python/notebook analysis on any supported device."""
    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text())
    refs = json.loads((run_dir / "artifacts.json").read_text())
    if run["experiment"] != experiment.name:
        raise ValueError("Wrong experiment class for this run.")
    runtime_config = {**run["config"]["runtime"], "device": device}
    context = Context(experiment, run["config"], run["member"],
                      Store(run_dir / refs["store_relative"]),
                      Runtime(runtime_config, initialize=False), run_dir)
    for name, relative in refs["items"].items():
        artifact = Artifact((run_dir / relative).resolve())
        artifact.verify()
        if name in {"training", "model"}:
            context.artifacts[name] = artifact
    return context


def verify_store(root: str | Path) -> int:
    """Recompute all saved artifact checksums; raises on the first damaged file."""
    count = 0
    for path in (Path(root) / "cache").glob("*/*/artifact.json"):
        Artifact(path.parent).verify(deep=True)
        count += 1
    return count
