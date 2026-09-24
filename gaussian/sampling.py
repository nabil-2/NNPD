"""Training sampling is independent of the model, optimizer and analysis hooks."""
from nnpd.core.settings import select
from nnpd.nre.data import balanced_pairs


def data_settings(context):
    return {**select(context.training_settings, "problem", "data", "seed"), "member": context.member,
            "prior": context.training_settings["priors"][context.member["prior"]]}


def training_data(context, writer):
    for split in ("train", "validation", "test"):
        x, theta, y = balanced_pairs(context.problem, context.prior,
                                     context.training_settings["data"][f"{split}_size"],
                                     context.rng(f"training-{split}"))
        for name, values in (("x", x), ("theta", theta), ("y", y)):
            writer.array(f"{split}_{name}", values)
    return {"parameters": context.problem.names, "class_one": "joint", "class_zero": "product",
            "split_policy": "independent, exactly balanced splits", "prior_measure": context.prior.measure}
