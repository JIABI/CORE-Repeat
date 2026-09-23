"""Portable wrapper for the paper's frozen CORE and direct-model inference.

Inputs must already use the exact frozen assay feature order and normalisation.
This is not an input adapter for a new plate, site or feature panel.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Trusted final model directory (contains mean/, distribution.joblib, histgb.joblib)")
    parser.add_argument("--query", type=Path, required=True,
                        help="Prepared NPZ with ids, groups, X, chem, chem_mask; optional layout")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; no previous prediction directory is overwritten")
    parser.add_argument("--population-size", type=int, required=True,
                        help="Full declared candidate count, including identities without an eligible X")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new output directory: " + str(args.output))
    import numpy as np
    from opal2.r4_final_model import load, score
    with np.load(args.query, allow_pickle=False) as saved:
        query = {key: saved[key].copy() for key in saved.files}
    # The original API validates identities, role isolation and the X-only schema.
    score(load(args.model), query, args.output, population_size=args.population_size)


if __name__ == "__main__":
    main()
