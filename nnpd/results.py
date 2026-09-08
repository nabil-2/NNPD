"""Small result readers: no numbered configuration IDs, pandas or hidden plotting state."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def runs(root, *, complete_only=True):
    records = []
    for path in sorted((Path(root) / "runs").glob("*/*/run.json")):
        status_path = path.with_name("status.json")
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if complete_only and status.get("state") != "complete":
            continue
        record = json.loads(path.read_text())
        metrics_path = path.with_name("metrics.json")
        record["metrics"] = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        record["path"] = str(path.parent)
        record["status"] = status
        records.append(record)
    return records


def value_at(value, path):
    """Dotted dictionary/list lookup, e.g. coverage.projected_mean.0."""
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def export_csv(root, destination, columns):
    """columns maps a CSV heading to a dotted path inside a complete run record."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_path", *columns])
        writer.writeheader()
        for record in runs(root):
            row = {"run_path": record["path"]}
            for name, path in columns.items():
                try:
                    value = value_at(record, path)
                except (KeyError, IndexError):
                    value = None
                row[name] = json.dumps(value) if isinstance(value, (dict, list)) else value
            writer.writerow(row)
    return destination


def comparison_plot(records, x_path, metric_path, *, group_path="member.prior"):
    """Return a figure; callers freely filter records and choose any numeric metric."""
    from matplotlib import pyplot as plt
    groups = {}
    for record in records:
        group = str(value_at(record, group_path))
        groups.setdefault(group, []).append((value_at(record, x_path), value_at(record, metric_path)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for group, values in sorted(groups.items()):
        values = sorted(values)
        ax.plot([x for x, _ in values], [y for _, y in values], marker="o", label=group)
    ax.set(xlabel=x_path, ylabel=metric_path)
    if groups:
        ax.legend()
    return fig
