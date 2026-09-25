"""Portable paths shared by the released publication plotting scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DATA_ROOT = Path(os.environ.get("OPAL2_DATA_ROOT", str(REPOSITORY.parent.parent / "zenodo" / "CORE-Repeat-data"))).expanduser().resolve()
DATA = DATA_ROOT / "source_data"
RESEARCH = DATA_ROOT / "research"
OUT = Path(os.environ.get("OPAL2_FIGURE_OUT", str(ROOT / "figures"))).expanduser().resolve()
QA = Path(os.environ.get("OPAL2_QA_OUT", str(ROOT / "qa"))).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
if not DATA.is_dir():
    raise FileNotFoundError("Set OPAL2_DATA_ROOT to the companion CORE-Repeat-data directory containing source_data/ and research/.")
