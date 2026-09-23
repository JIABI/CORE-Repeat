"""Export the implemented source, tests and lightweight recorded results.

The local datasets, environments and trained weights are intentionally not
redistributed by this source bundle. They remain in the experiment directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def export_source(root: Path, output: Path) -> int:
    root = root.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Choose a new source-bundle filename")
    files = set(root.glob("*.md"))
    files.update(root / name for name in ("pyproject.toml", "requirements-tested.txt"))
    for directory in ("opal2", "tests"):
        files.update((root / directory).rglob("*.py"))
    for directory in ("data/source5_primary_fullcontrols", "data/source5_spatial_fullcontrols"):
        for name in ("manifest.json", "splits.json"):
            path = root / directory / name
            if path.exists():
                files.add(path)
    for suffix in ("*.json", "*.tsv", "*.md"):
        files.update((root / "reports").rglob(suffix))
    files.update((root / "configs").glob("*.json"))
    permitted = {"evaluation.json", "config.json", "training.jsonl", "predictions.tsv",
                 "evaluation_predictions.tsv", "norm_diagnostics.tsv", "mc_precision.tsv",
                 "workflow.json", "decision_workflow.json", "repair_verification.json", "cross_validation.json",
                 "cross_validation_predictions.tsv", "run_manifest.json", "fold_manifest.json",
                 "allocations.tsv", "software_test_results.xml", "first_test_metrics.tsv",
                 "results_index.json", "INTERRUPTED.md"}
    for path in (root / "runs").rglob("*"):
        if path.is_file() and (path.name in permitted or path.name.startswith("probe_") and path.suffix == ".json"):
            files.add(path)
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Expected regular source/result file: {path}")
            archive.write(path, Path(root.name) / path.relative_to(root))
    with ZipFile(output) as archive:
        invalid = archive.testzip()
        if invalid is not None:
            raise IOError(f"Archive verification failed: {invalid}")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print({"output": str(args.output.resolve()), "files": export_source(args.root, args.output),
           "datasets_included": False, "trained_weights_included": False})
