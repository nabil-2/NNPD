"""Plan first; train and cache models; analyze saved models with separate analysis settings."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import platform
import shutil
import traceback
import uuid
import zipfile

import numpy as np

from .api import Analysis, Context, Experiment, implementation_hash, load_analysis, load_experiment
from .runtime import Runtime, seed_all
from .settings import at, default_profile, digest, expand_sweep, load_settings, single
from .storage import Artifact, Store, file_hash, safe_name, utc_now, write_json

TRAINING_FILE = "settings_training.py"
ANALYSIS_FILE = "settings_analysis.py"
KEEP_SETTINGS = 3  # the cache keeps the artifacts used by this many most recent training settings
STAGES = ("train", "run", "analyze")


@dataclass(frozen=True)
class Job:
    settings: dict
    member: dict
    changed: str | None
    workload: dict

    @property
    def id(self) -> str:
        return digest({"settings": self.settings, "member": self.member})

    def as_dict(self) -> dict:
        return {"id": self.id, "changed": self.changed, "member": self.member, "settings": self.settings}


def _validate_analysis(settings: dict, analysis: Analysis) -> None:
    products = analysis.products()
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
    for hooks in (analysis.metrics(), analysis.plots()):
        for name, hook in hooks.items():
            safe_name(name)
            for dependency in hook.needs:
                visit(dependency)
    analysis.validate(settings)
    for category, registry in (("metrics", analysis.metrics()), ("plots", analysis.plots())):
        unknown = set(settings[category]) - registry.keys()
        if unknown:
            raise ValueError(f"Unknown {category}: {sorted(unknown)}")
        if len(settings[category]) != len(set(settings[category])):
            raise ValueError(f"Duplicate names in {category}.")


def _launch_values(training_settings: dict) -> dict:
    """experiment, output and runtime are shared by every job of one launch."""
    variants = expand_sweep(training_settings)
    values = {key: variants[0].settings[key] for key in ("experiment", "output", "runtime")}
    for variant in variants[1:]:
        for key, value in values.items():
            if variant.settings[key] != value:
                raise ValueError(f"{key} cannot vary within one launch; use separate launches.")
    return values


def expand_jobs(training_settings: dict, experiment: Experiment,
                analysis_settings: dict | None = None, analysis: Analysis | None = None) -> list[Job]:
    """Validate *every* training alternative, and the analysis of each, before anything runs."""
    _launch_values(training_settings)
    if analysis is not None:
        analysis_settings = single(analysis_settings, "The analysis settings")
        _validate_analysis(analysis_settings, analysis)
    jobs = []
    for variant in expand_sweep(training_settings):
        settings = variant.settings
        experiment.validate(settings)
        members = experiment.members(settings)
        if not members:
            raise ValueError("An experiment must have at least one cohort member.")
        if len({digest(member) for member in members}) != len(members):
            raise ValueError("Duplicate cohort members.")
        for member in members:
            workload = dict(experiment.estimate(settings, member))
            if analysis is not None:
                problem = experiment.make_problem(settings)
                prior = experiment.make_prior(settings, member, problem)
                workload.update(analysis.estimate(analysis_settings, problem, prior))
            jobs.append(Job(settings, member, variant.changed, workload))
    return jobs


def infeasible(jobs: list[Job]) -> dict[str, list[str]]:
    """Configurations whose analysis would exceed the configured limits, with the reasons."""
    found: dict[str, list[str]] = {}
    for job in jobs:
        if job.workload.get("errors"):
            label = f"{job.changed}={at(job.settings, job.changed)!r}" if job.changed else "baseline"
            reasons = found.setdefault(label, [])
            reasons.extend(error for error in job.workload["errors"] if error not in reasons)
    return found


def check_feasible(jobs: list[Job]) -> None:
    """Refuse the whole launch before anything is computed if any configuration cannot be analyzed."""
    problems = infeasible(jobs)
    if problems:
        details = "\n".join(f"  {label}: {'; '.join(reasons)}" for label, reasons in problems.items())
        raise ValueError(f"Analysis is not feasible for these configurations; nothing was computed. "
                         f"Change the named settings:\n{details}")


@dataclass
class _Launch:
    """The settings files of one launch, loaded for one profile."""
    profile: str
    training_file: Path
    training: dict
    experiment: Experiment
    output: Path
    runtime: dict
    analysis_file: Path | None = None
    analysis_settings: dict | None = None
    analysis: Analysis | None = None


def _load(settings_dir: str | Path, profile: str | None, *, with_analysis: bool) -> _Launch:
    folder = Path(settings_dir)
    training_file = folder / TRAINING_FILE
    if not training_file.is_file():
        raise FileNotFoundError(f"{training_file} does not exist.")
    profile = profile or default_profile(training_file)
    training = load_settings(training_file, profile)
    values = _launch_values(training)
    launch = _Launch(profile, training_file, training, load_experiment(values["experiment"]),
                     Store(values["output"]).root, values["runtime"])
    if with_analysis:
        launch.analysis_file = folder / ANALYSIS_FILE
        if not launch.analysis_file.is_file():
            raise FileNotFoundError(f"{launch.analysis_file} does not exist.")
        launch.analysis_settings = single(load_settings(launch.analysis_file, profile), ANALYSIS_FILE)
        launch.analysis = load_analysis(launch.analysis_settings["analysis"])
    return launch


def plan(profile: str | None = None, settings_dir: str | Path = ".") -> list[Job]:
    """What `run` would train, with each model's analysis workload; computes nothing."""
    launch = _load(settings_dir, profile, with_analysis=True)
    return expand_jobs(launch.training, launch.experiment, launch.analysis_settings, launch.analysis)


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


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _source_snapshot(store: Store, classes: list[type], settings_files: list[Path]) -> str:
    """Snapshot the framework, the application packages, and the settings files used."""
    package = Path(__file__).resolve().parents[1]
    files = {f"nnpd/{path.relative_to(package)}": path for path in package.rglob("*.py")}
    for cls in classes:
        folder = Path(inspect.getfile(cls)).resolve().parent
        for path in folder.rglob("*.py"):
            files[f"{folder.name}/{path.relative_to(folder)}"] = path
        for name in ("run.py", "pyproject.toml"):
            path = folder.parent / name
            if path.exists():
                files[name] = path
    for path in settings_files:
        files[path.name] = Path(path).resolve()
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


def _train_model(context: Context) -> None:
    """Get or build the training data and the model; every DDP worker takes part."""
    experiment = context.experiment
    recipe = {"experiment": experiment.name, "stage": "training",
              "settings": experiment.signature("data", context),
              "implementation": implementation_hash(experiment.sources("data"), experiment.version)}
    context.artifacts["training"] = artifact = context.store.get(
        "training", digest(recipe), recipe, lambda writer: experiment.sample_training(context, writer))
    recipe = {"experiment": experiment.name, "stage": "model", "training": artifact.key,
              "settings": experiment.signature("model", context),
              "backend": context.runtime.training_signature,
              "implementation": implementation_hash(experiment.sources("model"), experiment.version)}
    key = digest(recipe)
    runtime = context.runtime
    lock = context.store.lock("model", key) if runtime.leader else nullcontext()
    with lock:
        cached = context.store.existing("model", key) if runtime.leader else None
        if not runtime.broadcast(cached is not None):
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


def _clear_results(run_dir: Path) -> None:
    (run_dir / "metrics.json").unlink(missing_ok=True)
    for name in ("metrics", "plots"):
        shutil.rmtree(run_dir / name, ignore_errors=True)


def analyze_run(context: Context) -> dict:
    """Compute the selected metrics and plots of one run, replacing earlier ones. Never trains."""
    settings = context.analysis_settings
    _clear_results(context.run_dir)
    metrics = {}
    for name in settings["metrics"]:
        hook = context.analysis.metrics()[name]
        value = hook.compute(context, context.dependencies(hook.needs))
        metrics[name] = _json_value(value)
        write_json(context.run_dir / "metrics" / f"{safe_name(name)}.json", metrics[name])
    if settings["plots"]:
        from matplotlib import pyplot as plt
        for name in settings["plots"]:
            hook = context.analysis.plots()[name]
            figures = hook.draw(context, context.dependencies(hook.needs))
            try:
                for stem, figure in figures.items():
                    path = context.run_dir / "plots" / safe_name(name) / f"{safe_name(stem)}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    figure.savefig(path, dpi=settings["plotting"]["dpi"], bbox_inches="tight")
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


def _run_dir(job: Job) -> Path:
    variant = (job.changed or "baseline").replace(".", "_")
    member = safe_name(job.member.get("name", "member"))
    return Store(job.settings["output"]).root / "runs" / f"{variant}-{digest(job.settings)[:12]}" / f"{member}-{job.id[:8]}"


def _open(run_dir: Path, runtime: Runtime, analysis: Analysis | None = None,
          analysis_settings: dict | None = None) -> Context:
    """A context for a trained run: its saved training settings, data and model."""
    run, references = _read_json(run_dir / "run.json"), _read_json(run_dir / "artifacts.json")
    experiment = load_experiment(run["settings"]["experiment"])
    if run["experiment"] != experiment.name:
        raise ValueError("Wrong experiment class for this run.")
    context = Context(experiment, run["settings"], run["member"], Store(run_dir / references["store_relative"]),
                      runtime, run_dir, analysis, analysis_settings)
    for name, relative in references["items"].items():
        artifact = Artifact((run_dir / relative).resolve())
        artifact.verify()
        if name in {"training", "model"}:
            context.artifacts[name] = artifact
    return context


def _trained_jobs(launch: _Launch) -> tuple[list[Job], list[Path]]:
    """The runs of the latest successful training in the output folder, with their analysis workload."""
    history = _read_json(launch.output / "history.json", [])
    if not history:
        raise FileNotFoundError(f"No trained models in {launch.output}; run 'train' or 'run' first. "
                                "Analysis never trains implicitly.")
    _validate_analysis(launch.analysis_settings, launch.analysis)
    jobs, run_dirs, experiments = [], [], {}
    for relative in history[0]["runs"]:
        run_dir = launch.output / relative
        state = _read_json(run_dir / "status.json", {}).get("state")
        if state != "trained":
            raise FileNotFoundError(f"{run_dir} is not trained ({state}); run 'train' again. "
                                    "Analysis never trains implicitly.")
        run = _read_json(run_dir / "run.json")
        spec = run["settings"]["experiment"]
        experiment = experiments.setdefault(spec, load_experiment(spec))
        problem = experiment.make_problem(run["settings"])
        prior = experiment.make_prior(run["settings"], run["member"], problem)
        workload = launch.analysis.estimate(launch.analysis_settings, problem, prior)
        jobs.append(Job(run["settings"], run["member"], run["changed"], workload))
        run_dirs.append(run_dir)
    return jobs, run_dirs


def _record_failure(path: Path, record: dict) -> None:
    write_json(path, {**record, "state": "failed", "time": utc_now(), "traceback": traceback.format_exc()})


def _train_run(context: Context, job: Job, source: str | None) -> None:
    run_dir, runtime = context.run_dir, context.runtime
    if runtime.leader:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis.json").unlink(missing_ok=True)  # new training: earlier analysis no longer applies
        _clear_results(run_dir)
        write_json(run_dir / "run.json", {**job.as_dict(), "experiment": context.experiment.name,
                   "source_archive": source, "python": platform.python_version(),
                   "backend": runtime.training_signature, "numpy": np.__version__, "started": utc_now()})
        write_json(run_dir / "status.json", {"state": "running", "time": utc_now()})
    try:
        _train_model(context)
    except Exception:
        if runtime.leader:
            _record_failure(run_dir / "status.json", {})
        raise
    if runtime.leader:
        save_references(context)
        write_json(run_dir / "status.json", {"state": "trained", "time": utc_now()})


def _analyze(context: Context, source: str) -> None:
    record = {"analysis": context.analysis.name, "settings": context.analysis_settings,
              "source_archive": source, "started": utc_now()}
    write_json(context.run_dir / "analysis.json", {**record, "state": "running"})
    try:
        analyze_run(context)
    except Exception:
        _record_failure(context.run_dir / "analysis.json", record)
        raise
    write_json(context.run_dir / "analysis.json", {**record, "state": "complete", "time": utc_now()})


def _finish_launch(launch: _Launch, run_dirs: list[Path], stage: str, launch_id: str) -> None:
    """Latest settings win: keep only this launch's runs, and the cache of the last KEEP_SETTINGS trainings.

    The settings files the launch started from are copied byte for byte into the output folder.
    Only the process that finishes a launch last does this, so a failed or unfinished launch deletes nothing.
    """
    root = launch.output
    with Store(root).lock("experiment", "history"):
        marks = [_read_json(path / "status.json", {}) for path in run_dirs]
        if any(mark.get("launch") != launch_id for mark in marks):
            return  # other workers of this launch are still running; the last one cleans up
        used = set()
        for path in run_dirs:
            for relative in _read_json(path / "artifacts.json")["items"].values():
                used.add((path / relative).resolve().relative_to(root).as_posix())
            for record in ("run.json", "analysis.json"):
                source = _read_json(path / record, {}).get("source_archive")
                if source:
                    used.add(f"source/{source}.zip")
        runs = sorted(path.relative_to(root).as_posix() for path in run_dirs)
        version = digest(runs)
        history = _read_json(root / "history.json", [])
        previous = next((entry for entry in history if entry["version"] == version), None)
        if stage == "train" and previous:
            # The same training again: keep its analysis data for the next analyze.
            used |= {item for item in previous["uses"] if not item.startswith(("cache/training/", "cache/model/"))}
        history = [{"version": version, "time": utc_now(), "stage": stage, "runs": runs, "uses": sorted(used)},
                   *(entry for entry in history if entry["version"] != version)][:KEEP_SETTINGS]
        write_json(root / "history.json", history)
        files = {"train": [launch.training_file], "analyze": [launch.analysis_file],
                 "run": [launch.training_file, launch.analysis_file]}[stage]
        for path in files:
            if path.resolve() != (root / path.name).resolve():
                shutil.copyfile(path, root / path.name)
        keep = {item for entry in history for item in entry["uses"]}
        current = set(run_dirs)
        for path in (root / "runs").glob("*/*"):
            if path not in current:
                shutil.rmtree(path)
        for path in (root / "runs").glob("*"):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        for path in (root / "cache").glob("*/*"):
            # Names starting with "." are builds in progress, never committed artifacts.
            if not path.name.startswith(".") and path.relative_to(root).as_posix() not in keep:
                shutil.rmtree(path)
                (root / "locks" / f"{path.parent.name}-{path.name}.lock").unlink(missing_ok=True)
        for path in (root / "source").glob("*.zip"):
            if path.relative_to(root).as_posix() not in keep:
                path.unlink()


def execute(stage: str = "run", *, profile: str | None = None, settings_dir: str | Path = ".") -> list[Path]:
    """Train, analyze, or both, from the settings files in settings_dir; the CLI does the same.

    'train' trains the models of settings_training.py and resets their analysis.
    'analyze' analyzes the latest trained models with settings_analysis.py; it never trains.
    'run' does both. 'jobs': torchrun workers process disjoint runs; 'ddp': workers train
    each model together and only rank zero analyzes. After success, runs of earlier settings
    are removed, the cache keeps what the last KEEP_SETTINGS trainings used, and the settings
    files are copied into the output folder.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}.")
    launch = _load(settings_dir, profile, with_analysis=stage != "train")
    if stage == "analyze":
        jobs, run_dirs = _trained_jobs(launch)
    else:
        jobs = expand_jobs(launch.training, launch.experiment, launch.analysis_settings, launch.analysis)
        run_dirs = [_run_dir(job) for job in jobs]
    if stage != "train":
        check_feasible(jobs)
    runtime = Runtime(launch.runtime)
    store = Store(launch.output)
    # torchrun gives all workers of one launch the same run ID; a single process makes its own.
    launch_id = os.environ.get("TORCHELASTIC_RUN_ID") or uuid.uuid4().hex
    training_source = analysis_source = None
    completed = []
    try:
        for index, (job, run_dir) in enumerate(zip(jobs, run_dirs)):
            if not runtime.ddp and index % runtime.world_size != runtime.rank:
                continue
            if runtime.leader:
                if stage != "analyze" and training_source is None:
                    training_source = _source_snapshot(store, [type(launch.experiment)], [launch.training_file])
                if stage != "train" and analysis_source is None:
                    analysis_source = _source_snapshot(store, [type(launch.analysis)], [launch.analysis_file])
                variant = (job.changed or "baseline").replace(".", "_")
                print(f"[{index + 1}/{len(jobs)}] {stage} {variant}: {job.member.get('name', 'member')}", flush=True)
            if stage == "analyze":
                context = _open(run_dir, runtime, launch.analysis, launch.analysis_settings)
            else:
                context = Context(launch.experiment, job.settings, job.member, store, runtime, run_dir,
                                  launch.analysis, launch.analysis_settings)
                _train_run(context, job, training_source)
            if runtime.leader:
                if stage != "train":
                    _analyze(context, analysis_source)
                # Marks this run as finished by this launch; the last worker to finish cleans up.
                write_json(run_dir / "status.json", {**_read_json(run_dir / "status.json"), "launch": launch_id})
                completed.append(run_dir)
            runtime.barrier()
        if runtime.leader:
            _finish_launch(launch, run_dirs, stage, launch_id)
    finally:
        runtime.close()
    return completed


def restore(run_dir: str | Path, *, device: str = "cpu") -> Context:
    """Open a saved run, with the analysis settings it was analyzed with, on any supported device."""
    run_dir = Path(run_dir).resolve()
    run = _read_json(run_dir / "run.json")
    record = _read_json(run_dir / "analysis.json")
    analysis = load_analysis(record["settings"]["analysis"]) if record else None
    runtime = Runtime({**run["settings"]["runtime"], "device": device}, initialize=False)
    return _open(run_dir, runtime, analysis, record["settings"] if record else None)


def verify_store(root: str | Path) -> int:
    """Recompute all saved artifact checksums; raises on the first damaged file."""
    count = 0
    for path in (Path(root) / "cache").glob("*/*/artifact.json"):
        Artifact(path.parent).verify(deep=True)
        count += 1
    return count
