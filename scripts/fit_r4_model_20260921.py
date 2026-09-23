"""Fit the final R4 models from the already opened EU904 development dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opal2.r4_final_model import fit, SEED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--partitions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    fit(args.dataset, args.partitions, args.output, seed=args.seed)


if __name__ == '__main__':
    main()
