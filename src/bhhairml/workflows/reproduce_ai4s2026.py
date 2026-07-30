"""Regenerate the AI4S 2026 results, figures, claims, and paper workspace."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn

from bhhairml.experiments.geodesic_extension import run as run_geodesic
from bhhairml.experiments.kiselev_identifiability import run as run_identifiability
from bhhairml.experiments.waveform_to_hair import run as run_waveform
from bhhairml.plots import save_figure
from bhhairml.plots.poster_physics_figures import kiselev_identifiability_figure
from bhhairml.results import collect_ai4s2026_claims, validate_claims
from bhhairml.utils.io import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _compile_paper(paper_dir, required=False):
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    tectonic = shutil.which("tectonic")
    if pdflatex and bibtex:
        commands = [
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [bibtex, "main"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ]
    elif tectonic:
        commands = [[tectonic, "--keep-logs", "main.tex"]]
    else:
        message = "LaTeX unavailable; source workspace generated without PDF"
        if required:
            raise RuntimeError(message)
        return {"built": False, "message": message}
    logs = []
    for command in commands:
        result = subprocess.run(
            command, cwd=paper_dir, text=True, capture_output=True)
        logs.append({"command": command, "returncode": result.returncode,
                     "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            (paper_dir / "build_failure.json").write_text(
                json.dumps(logs, indent=2), encoding="utf-8")
            raise RuntimeError(f"Paper build failed: {' '.join(command)}")
    return {"built": True, "pdf": str(paper_dir / "main.pdf")}


def run(config_path, *, output_root=None, require_latex=False,
        update_claims=False):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml(config_path)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(output_root or config["artifact_root"])
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    root = base / stamp
    root.mkdir(parents=True, exist_ok=False)

    ident_config = PROJECT_ROOT / config["identifiability_config"]
    geodesic_config = PROJECT_ROOT / config["geodesic_config"]
    waveform_config = PROJECT_ROOT / config["waveform_config"]

    run_identifiability(str(ident_config), output_root=root)
    run_geodesic(
        str(ident_config), str(geodesic_config), output_root=root)
    run_waveform(str(waveform_config), output_root=root)

    figure, error = kiselev_identifiability_figure(root / "tables")
    if figure is None:
        raise RuntimeError(error)
    paper_figure_dir = root / "paper" / "figures"
    paper_figure_dir.mkdir(parents=True)
    save_figure(
        figure, paper_figure_dir / "kiselev_identifiability_lens")
    plt.close(figure)

    waveform_figures = root / "figures" / "poster"
    for name in (
        "learning_when_hair_is_observable",
        "waveform_to_hair_feature_comparison",
        "waveform_to_hair_true_vs_pred",
    ):
        destination_name = (
            "learning_when_hair_is_observable_rank_aware"
            if name == "learning_when_hair_is_observable" else name)
        shutil.copy2(
            waveform_figures / f"{name}.pdf",
            paper_figure_dir / f"{destination_name}.pdf")

    paper_source = PROJECT_ROOT / config["paper_source"]
    shutil.copy2(paper_source / "main.tex", root / "paper" / "main.tex")
    shutil.copy2(
        paper_source / "references.bib",
        root / "paper" / "references.bib")

    claims = collect_ai4s2026_claims(root)
    claims_path = root / "claims.json"
    claims_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    expected_path = PROJECT_ROOT / config["claims_manifest"]
    if update_claims:
        expected_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
        validation = []
    else:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        validation = validate_claims(claims, expected)
    (root / "claim_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8")

    build = _compile_paper(
        root / "paper",
        required=require_latex or bool(config.get("require_latex", False)))
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "python": sys.version,
        "dependencies": {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "config": str(config_path),
        "paper_build": build,
        "files": {
            str(path.relative_to(root)).replace("\\", "/"): _sha256(path)
            for path in files
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"AI4S 2026 reproduction complete: {root}")
    print(f"Claims: {claims_path}")
    print(build["message"] if not build["built"] else build["pdf"])
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ai4s2026.yaml")
    parser.add_argument("--output-root")
    parser.add_argument("--require-latex", action="store_true")
    parser.add_argument("--update-claims", action="store_true")
    args = parser.parse_args()
    run(args.config, output_root=args.output_root,
        require_latex=args.require_latex, update_claims=args.update_claims)


if __name__ == "__main__":
    main()
