from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from nnpd import Metric, Product, plan
from nnpd.core.api import Context, implementation_hash
from nnpd.core.runtime import Runtime
from nnpd.core.storage import Artifact, Store, safe_name
from nnpd.core.runner import verify_store
from gaussian import GaussianExperiment


def first_builder(context, dependencies, writer):
    writer.array("x", np.arange(4))
    return {"kind": "first"}


def second_builder(context, dependencies, writer):
    writer.array("x", np.arange(5))
    return {"kind": "second"}


def test_fingerprints_distinguish_different_functions_in_same_module():
    assert implementation_hash((first_builder,), "1") != implementation_hash((second_builder,), "1")
    assert implementation_hash((first_builder,), "1") != implementation_hash((first_builder,), "2")


def test_atomic_failure_never_becomes_cache_hit(tmp_path):
    store = Store(tmp_path)
    def fail(writer):
        writer.array("unfinished", np.arange(5))
        raise RuntimeError("injected interruption")
    with pytest.raises(RuntimeError, match="interruption"):
        store.get("data", "abc", {}, fail)
    assert store.existing("data", "abc") is None
    assert list((tmp_path / "cache" / "data").iterdir()) == []
    with pytest.raises(RuntimeError, match="finalize"):
        with store.transaction("data", "abc", {}) as writer:
            writer.array("unfinished", np.arange(5))


def test_array_roundtrip_memmap_and_corruption_detection(tmp_path):
    store = Store(tmp_path)
    def build(writer):
        values = writer.allocate("large", (20, 3), dtype="float64")
        values[:] = np.arange(60).reshape(20, 3)
        writer.json("notes", {"reason": "test"})
        return {"count": 20}
    artifact = store.get("data", "abc", {"seed": 1}, build)
    assert isinstance(artifact.array("large"), np.memmap)
    np.testing.assert_equal(artifact.array("large"), np.arange(60).reshape(20, 3))
    assert artifact.json("notes") == {"reason": "test"}
    assert artifact.metadata == {"count": 20}
    assert verify_store(tmp_path) == 1
    path = artifact.path / "large.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        old = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([old[0] ^ 1]))
    artifact.verify()  # fast default checks size; deep checks detect same-size corruption
    with pytest.raises(IOError, match="Checksum"):
        artifact.verify(deep=True)
    path.unlink()
    with pytest.raises(IOError, match="damaged"):
        store.existing("data", "abc")


@pytest.mark.parametrize("name", ["../foo", "/tmp/foo", "a/b", "..", ""])
def test_unsafe_names_rejected(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_object_arrays_rejected(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(TypeError, match="Object"):
        store.get("data", "abc", {}, lambda writer: writer.array("x", np.array([{}], dtype=object)))


class ProductsOnly(GaussianExperiment):
    entries = {}
    def products(self):
        return self.entries
    def metrics(self):
        return {"unused": Metric(lambda context, dependencies: 0)}
    def plots(self):
        return {}


def context_for(tiny, experiment):
    member = experiment.members(tiny)[0]
    return Context(experiment, tiny, member, Store(tiny["output"]), Runtime(tiny["runtime"]), Path(tiny["output"]) / "run")


def test_shared_product_builds_once_and_key_controls_reuse(tiny):
    experiment = ProductsOnly()
    experiment.entries = {"shared": Product(first_builder, settings=lambda context: {"seed": context.config["seed"]})}
    context = context_for(tiny, experiment)
    a = context.require("shared")
    assert context.require("shared") is a
    stamp = (a.path / "x.npy").stat().st_mtime_ns
    other = context_for({**tiny, "metrics": ["anything"]}, experiment)
    assert other.require("shared").path == a.path
    assert (a.path / "x.npy").stat().st_mtime_ns == stamp
    changed = context_for({**tiny, "seed": tiny["seed"] + 1}, experiment)
    assert changed.require("shared").path != a.path
    experiment.entries["shared"] = replace(experiment.entries["shared"], version="2")
    assert context_for(tiny, experiment).require("shared").path != a.path
    experiment.entries["shared"] = Product(second_builder, settings=lambda context: {"seed": context.config["seed"]})
    assert len(context_for(tiny, experiment).require("shared").array("x")) == 5


def test_dependency_cycle_and_missing_names_fail_before_training(tiny):
    tiny["metrics"], tiny["plots"] = [], []
    experiment = ProductsOnly()
    experiment.entries = {"a": Product(first_builder, ("b",)), "b": Product(first_builder, ("a",))}
    with pytest.raises(ValueError, match="cycle"):
        plan(tiny, experiment)
    with pytest.raises(ValueError, match="cycle"):
        context_for(tiny, experiment).require("a")
    experiment.entries = {"a": Product(first_builder, ("missing",))}
    with pytest.raises(KeyError, match="missing"):
        plan(tiny, experiment)
    experiment.entries = {"model": Product(first_builder)}
    with pytest.raises(ValueError, match="reserved"):
        plan(tiny, experiment)
