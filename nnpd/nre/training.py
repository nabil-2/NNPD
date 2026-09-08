"""Readable BCE training, including exact global-batch sharding for DDP."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F


def predict_logits(model, x, theta, batch_size: int, device) -> np.ndarray:
    x, theta = np.asarray(x), np.asarray(theta)
    if x.ndim != 2 or theta.ndim not in {1, 2}:
        raise ValueError("x must be a matrix and theta a vector or matrix.")
    if theta.ndim == 1:
        theta = np.broadcast_to(theta, (len(x), len(theta)))
    if len(x) != len(theta) or batch_size < 1:
        raise ValueError("Mismatched sample counts or invalid prediction batch size.")
    output = np.empty(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            inputs = np.concatenate((x[start:stop], theta[start:stop]), axis=1).astype(np.float32)
            logits = model(torch.from_numpy(inputs).to(device)).reshape(-1)
            output[start:stop] = logits.double().cpu().numpy()
    if not np.isfinite(output).all():
        raise FloatingPointError("The model returned nonfinite logits; refusing invalid ratio estimates.")
    return output


def classification(labels: np.ndarray, logits: np.ndarray) -> dict:
    from scipy.stats import rankdata
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    positives = labels == 1
    n1, n0 = int(positives.sum()), int((labels == 0).sum())
    if n1 + n0 != len(labels) or not n1 or not n0:
        raise ValueError("Classification metrics need both binary classes.")
    ranks = rankdata(logits, method="average")
    auc = (ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"bce": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
            "accuracy": float(np.mean((logits >= 0) == positives)), "auc": float(auc)}


def fit_binary(context, model) -> dict:
    """No sample padding or dropping: every training row contributes once per epoch.

    A batch is split by rank. Local loss SUM * world/global_count compensates
    for DDP's gradient averaging even for unequal or empty last shards.
    An empty shard runs a zero-weight dummy forward to participate in collectives.
    The configured batch size is GLOBAL, not per GPU.
    """
    cfg, runtime = context.config["training"], context.runtime
    data = context.artifacts["training"]
    x, theta, labels = (data.array("train_" + name) for name in ("x", "theta", "y"))
    global_batch = cfg["batch_size"]
    model.to(runtime.device)
    train_model = DistributedDataParallel(model, device_ids=[runtime.device.index]
                                        if runtime.device.type == "cuda" else None) if runtime.ddp else model
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg["optimizer"]]
    optimizer = optimizer_class(train_model.parameters(), lr=cfg["learning_rate"],
                                betas=tuple(cfg["betas"]), eps=cfg["epsilon"],
                                weight_decay=cfg["weight_decay"])
    amp = cfg["precision"]
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(amp)
    if amp == "float16" and runtime.device.type != "cuda":
        raise ValueError("float16 training requires CUDA; use off or bfloat16 on CPU.")
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "float16")
    world = runtime.world_size if runtime.ddp else 1
    rank = runtime.rank if runtime.ddp else 0
    history = {"train_loss": [], "validation": [], "rows_per_epoch": len(x),
               "global_batch_size": global_batch, "world_size": world, "selected_epoch": None}
    best_loss, best_state = float("inf"), None
    for epoch in range(cfg["epochs"]):
        train_model.train()
        order = context.rng(f"epoch-{epoch}").permutation(len(x))
        loss_total = torch.zeros((), device=runtime.device, dtype=torch.float64)
        for start in range(0, len(x), global_batch):
            indices = order[start:start + global_batch]
            local = indices[rank::world]
            weight = 1.0 if len(local) else 0.0
            if not len(local):
                local = indices[:1]
            features = np.concatenate((x[local], theta[local]), axis=1)
            inputs = torch.from_numpy(features).to(runtime.device)
            target = torch.from_numpy(np.array(labels[local])).to(runtime.device)
            optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast(runtime.device.type, dtype=dtype) if dtype else nullcontext()
            with autocast:
                logits = train_model(inputs).reshape(-1)
                local_sum = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") * weight
                loss = local_sum * world / len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss.")
            scaler.scale(loss).backward()
            if cfg["gradient_clip"] is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg["gradient_clip"],
                                               error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            loss_total += local_sum.detach().double()
        if runtime.ddp:
            dist.all_reduce(loss_total)
        validation = None
        if runtime.leader:
            val_logits = predict_logits(model, data.array("validation_x"), data.array("validation_theta"),
                                        cfg["evaluation_batch_size"], runtime.device)
            validation = classification(data.array("validation_y"), val_logits)
        validation = runtime.broadcast(validation)
        history["train_loss"].append(float(loss_total.item() / len(x)))
        history["validation"].append(validation)
        if validation["bce"] < best_loss:
            best_loss = validation["bce"]
            if cfg["checkpoint"] == "best":
                best_state = deepcopy(model.state_dict())
                history["selected_epoch"] = epoch + 1
        if cfg["verbose"] and runtime.leader:
            print(f"  epoch {epoch + 1}: train={history['train_loss'][-1]:.5f}, val={validation['bce']:.5f}",
                  flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if cfg["checkpoint"] == "last":
        history["selected_epoch"] = cfg["epochs"]
    return history
