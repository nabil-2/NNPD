#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from os.path import relpath
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO_ROOT / "paper_latex-code"
GENERATED_TEX = PAPER_DIR / "plot_collection.tex"
DEFAULT_OUTPUT = REPO_ROOT / "figs" / "plot_collection.pdf"

CONFIG_DIMENSIONS = OrderedDict(
    [
        ("1D", "config_018"),
        ("2D", "config_013"),
        ("3D", "config_012"),
        ("4D", "config_014"),
        ("5D", "config_015"),
        ("6D", "config_017"),
    ]
)

CONFIG_ORDER = OrderedDict(
    [
        (
            "training",
            [
                "training/all_roc_curves.pdf",
                "training/training_0/training_exponential_prior.pdf",
                "training/training_0/training_grid_prior.pdf",
                "training/training_0/training_normal_prior.pdf",
                "training/training_0/training_uniform_prior.pdf",
            ],
        ),
        (
            "posterior_errors",
            [
                "posterior_errors/error_and_hpd_width.pdf",
                "posterior_errors/error_and_hpd_width_ratios.pdf",
                "posterior_errors/errorbars.pdf",
                "posterior_errors/errorbars_ratios.pdf",
            ],
        ),
        (
            "verifications",
            [
                "verifications/log_ratio_error_vs_distance.pdf",
                "verifications/log_ratio_exact_vs_predicted.pdf",
                "verifications/ratio_violins.pdf",
                "verifications/reweighting_0/reweighted_distributions_3.0_7.0.pdf",
                "verifications/reweighting_summary.pdf",
                "verifications/reweighting_swd_vs_distance.pdf",
            ],
        ),
    ]
)


@dataclass(frozen=True)
class DividerPage:
    title: str
    subtitle: str


@dataclass(frozen=True)
class PlotPage:
    section: str
    title: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a landscape PDF collection of selected NNPD plots."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the final compiled PDF.",
    )
    parser.add_argument(
        "--shared-priors",
        default="config_013",
        help="Config folder to source shared priors from, e.g. config_013 or 013.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered manifest and expected page counts without compiling.",
    )
    return parser.parse_args()


def normalize_config_name(value: str) -> str:
    value = value.strip()
    if value.startswith("config_"):
        return value
    if value.isdigit():
        return f"config_{value.zfill(3)}"
    raise ValueError(f"Unsupported config name: {value}")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def humanize_name(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    humanized = []
    for part in parts:
        upper = part.upper()
        if upper == "HPD":
            humanized.append("HLD")
        elif upper in {"ROC", "RMSE", "ESS", "SWD", "HLD", "MAP"}:
            humanized.append(upper)
        elif part.endswith("d") and part[:-1].isdigit():
            humanized.append(part[:-1] + "D")
        elif part.replace(".", "", 1).isdigit():
            humanized.append(part)
        else:
            humanized.append(part.capitalize())
    return " ".join(humanized)


def stable_pdf_listing(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pdf"), key=lambda path: path.name)


def validate_paths(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = "\n".join(f" - {path.relative_to(REPO_ROOT)}" for path in missing[:10])
        raise FileNotFoundError(f"Missing expected PDF inputs:\n{preview}")


def build_manifest(shared_priors_config: str) -> list[DividerPage | PlotPage]:
    manifest: list[DividerPage | PlotPage] = []

    manifest.append(
        DividerPage(
            "NNPD Plot Collection",
            "Shared priors, config sweeps, dimensionality analysis, and prior analysis",
        )
    )

    priors_dir = REPO_ROOT / "figs" / shared_priors_config / "priors"
    shared_priors = [
        priors_dir / "prior_contours.pdf",
        priors_dir / "prior_sample_histograms.pdf",
    ]
    validate_paths(shared_priors)
    manifest.append(DividerPage("Shared Priors", f"Representative prior plots from {shared_priors_config}"))
    manifest.extend(
        PlotPage("Shared Priors", humanize_name(path), path) for path in shared_priors
    )

    manifest.append(DividerPage("Config Results", "Successful runs from 1D through 6D"))
    for label, config_name in CONFIG_DIMENSIONS.items():
        manifest.append(DividerPage(label, config_name))
        config_root = REPO_ROOT / "figs" / config_name
        paths = [config_root / rel_path for group in CONFIG_ORDER.values() for rel_path in group]
        validate_paths(paths)
        manifest.extend(
            PlotPage(f"Config Results / {label}", humanize_name(path), path) for path in paths
        )

    dimensionality_dir = REPO_ROOT / "figs" / "dimensionality_analysis"
    dimensionality_paths = stable_pdf_listing(dimensionality_dir)
    validate_paths(dimensionality_paths)
    manifest.append(DividerPage("Dimensionality Analysis", "Cross-dimension aggregate metrics"))
    manifest.extend(
        PlotPage("Dimensionality Analysis", humanize_name(path), path)
        for path in dimensionality_paths
    )

    prior_analysis_dir = REPO_ROOT / "figs" / "prior_analysis"
    prior_analysis_paths = stable_pdf_listing(prior_analysis_dir)
    validate_paths(prior_analysis_paths)
    manifest.append(DividerPage("Prior Analysis", "Cross-prior aggregate metrics"))
    manifest.extend(
        PlotPage("Prior Analysis", humanize_name(path), path) for path in prior_analysis_paths
    )

    return manifest


def summarize_manifest(manifest: list[DividerPage | PlotPage]) -> str:
    counts: OrderedDict[str, int] = OrderedDict()
    lines = []

    for item in manifest:
        if isinstance(item, DividerPage):
            lines.append(f"[divider] {item.title} :: {item.subtitle}")
            continue

        counts[item.section] = counts.get(item.section, 0) + 1
        rel_path = item.source.relative_to(REPO_ROOT).as_posix()
        lines.append(f"[plot] {item.section} :: {item.title} :: {rel_path}")

    divider_count = sum(isinstance(item, DividerPage) for item in manifest)
    plot_count = len(manifest) - divider_count

    summary_lines = ["Manifest summary:"]
    for section, count in counts.items():
        summary_lines.append(f" - {section}: {count}")
    summary_lines.append(f" - divider pages: {divider_count}")
    summary_lines.append(f" - plot pages: {plot_count}")
    summary_lines.append(f" - total pages: {len(manifest)}")
    summary_lines.append("")
    summary_lines.append("Ordered manifest:")
    summary_lines.extend(f" - {line}" for line in lines)
    return "\n".join(summary_lines)


def tex_preamble() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[paperwidth=11in,paperheight=8.5in,margin=0.45in]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\definecolor{DividerAccent}{RGB}{25,68,120}
\definecolor{DividerMuted}{RGB}{90,98,108}
\definecolor{HeaderMuted}{RGB}{96,96,96}

\newcommand{\dividerpage}[2]{
  \clearpage
  \thispagestyle{empty}
  \vspace*{\fill}
  {\centering
  {\fontsize{28}{34}\selectfont\bfseries\color{DividerAccent} #1\par}
  \vspace{0.8cm}
  {\Large\color{DividerMuted} #2\par}
  }
  \vspace*{\fill}
}

\newcommand{\plotpage}[3]{
  \clearpage
  \thispagestyle{empty}
  {\large\bfseries #1\par}
  \vspace{0.1cm}
  {\LARGE\bfseries #2\par}
  \vspace{0.15cm}
  {\small\color{HeaderMuted}\texttt{\detokenize{#3}}\par}
  \vspace{0.25cm}
  \begin{center}
    \includegraphics[width=\textwidth,height=0.79\textheight,keepaspectratio]{#3}
  \end{center}
}

\begin{document}
"""


def render_tex(manifest: list[DividerPage | PlotPage]) -> str:
    chunks = [tex_preamble()]
    for item in manifest:
        if isinstance(item, DividerPage):
            chunks.append(
                "\\dividerpage"
                f"{{{latex_escape(item.title)}}}"
                f"{{{latex_escape(item.subtitle)}}}\n"
            )
            continue

        tex_source = relpath(item.source, PAPER_DIR).replace("\\", "/")
        chunks.append(
            "\\plotpage"
            f"{{{latex_escape(item.section)}}}"
            f"{{{latex_escape(item.title)}}}"
            f"{{{tex_source}}}\n"
        )

    chunks.append("\\end{document}\n")
    return "".join(chunks)


def run_pdflatex(tex_path: Path) -> Path:
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    subprocess.run(command, cwd=tex_path.parent, check=True)
    return tex_path.with_suffix(".pdf")


def build_collection(manifest: list[DividerPage | PlotPage], output_path: Path) -> Path:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_TEX.write_text(render_tex(manifest), encoding="utf-8")
    built_pdf = run_pdflatex(GENERATED_TEX)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_pdf, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    shared_priors_config = normalize_config_name(args.shared_priors)
    manifest = build_manifest(shared_priors_config)

    if args.dry_run:
        print(summarize_manifest(manifest))
        return

    output_path = args.output.resolve()
    built_path = build_collection(manifest, output_path)
    print(f"Built landscape plot collection at {built_path}")


if __name__ == "__main__":
    main()
