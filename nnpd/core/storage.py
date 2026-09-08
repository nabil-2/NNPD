"""Atomic, relocatable artifacts: JSON metadata, numeric NPY arrays, state dicts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Callable, Iterator

from filelock import FileLock
import numpy as np
import torch

from .config import canonical


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Unsafe artifact name: {name!r}")
    return name


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomic replacement, also for small run-status files outside artifacts."""
    text = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        f.write(json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n")
        temporary = Path(f.name)
    temporary.replace(path)


class Artifact:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "artifact.json").read_text())
        self.key = self.manifest["key"]
        self.metadata = self.manifest["metadata"]

    def array(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.path / f"{safe_name(name)}.npy", mmap_mode="r" if mmap else None,
                       allow_pickle=False)

    def json(self, name: str) -> object:
        return json.loads((self.path / f"{safe_name(name)}.json").read_text())

    def state_dict(self) -> dict:
        return torch.load(self.path / "weights.pt", map_location="cpu", weights_only=True)

    def verify(self, deep: bool = False) -> None:
        for name, info in self.manifest["files"].items():
            path = self.path / safe_name(name)
            if not path.is_file() or path.stat().st_size != info["bytes"]:
                raise IOError(f"Incomplete or damaged artifact: {path}")
            if deep and file_hash(path) != info["sha256"]:
                raise IOError(f"Checksum mismatch: {path}")


class Writer:
    """Passed to producers. Large products can allocate and fill arrays in chunks."""
    def __init__(self, path: Path):
        self.path = path
        self._arrays: list[np.memmap] = []

    def array(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values)
        if values.dtype.hasobject:
            raise TypeError("Object arrays are not supported; use JSON for metadata.")
        np.save(self.path / f"{safe_name(name)}.npy", values, allow_pickle=False)

    def allocate(self, name: str, shape: tuple[int, ...], dtype="float32") -> np.memmap:
        if np.dtype(dtype).hasobject:
            raise TypeError("Object arrays are not supported.")
        array = np.lib.format.open_memmap(self.path / f"{safe_name(name)}.npy", mode="w+",
                                         dtype=dtype, shape=shape)
        self._arrays.append(array)
        return array

    def json(self, name: str, value: object) -> None:
        if name == "artifact":
            raise ValueError("'artifact' is a reserved metadata name.")
        write_json(self.path / f"{safe_name(name)}.json", value)

    def weights(self, model: torch.nn.Module) -> None:
        torch.save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                   self.path / "weights.pt")

    def finish(self, name: str, key: str, metadata: dict, recipe: dict) -> None:
        for array in self._arrays:
            array.flush()
        self._arrays.clear()
        files = {}
        for path in sorted(self.path.iterdir()):
            info = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
            if path.suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                info.update(shape=list(array.shape), dtype=str(array.dtype))
            files[path.name] = info
        write_json(self.path / "artifact.json", {
            "schema": 1, "name": name, "key": key, "created": utc_now(),
            "metadata": metadata, "recipe": recipe, "files": files,
        })


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def path(self, name: str, key: str) -> Path:
        return self.root / "cache" / safe_name(name) / safe_name(key)

    def lock(self, name: str, key: str) -> FileLock:
        folder = self.root / "locks"
        folder.mkdir(parents=True, exist_ok=True)
        return FileLock(folder / f"{safe_name(name)}-{safe_name(key)}.lock")

    def existing(self, name: str, key: str) -> Artifact | None:
        path = self.path(name, key)
        if not (path / "artifact.json").is_file():
            return None
        artifact = Artifact(path)
        artifact.verify()
        return artifact

    @contextmanager
    def transaction(self, name: str, key: str, recipe: dict) -> Iterator[Writer]:
        """Caller holds the lock. A failed build never becomes a cache hit."""
        destination = self.path(name, key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=destination.parent))
        writer = Writer(temporary)
        try:
            yield writer
            if not (temporary / "artifact.json").is_file():
                raise RuntimeError("Producer did not finalize its artifact.")
            if destination.exists():
                raise FileExistsError(destination)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def get(self, name: str, key: str, recipe: dict,
            build: Callable[[Writer], dict]) -> Artifact:
        with self.lock(name, key):
            artifact = self.existing(name, key)
            if artifact is None:
                with self.transaction(name, key, recipe) as writer:
                    metadata = build(writer)
                    writer.finish(name, key, metadata, recipe)
                artifact = self.existing(name, key)
            assert artifact is not None
            return artifact
