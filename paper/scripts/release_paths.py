"""Portable paths shared by the released publication plotting scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DATA_ROOT = Path(os.environ.get("OPAL2_DATA_ROOT", str(REPOSITORY.parent / "zenodo_data"))).expanduser().resolve()
DATA = DATA_ROOT / "source_data"
RESEARCH = DATA_ROOT / "research"
OUT = Path(os.environ.get("OPAL2_FIGURE_OUT", str(ROOT / "figures"))).expanduser().resolve()
QA = Path(os.environ.get("OPAL2_QA_OUT", str(ROOT / "qa"))).expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
QA.mkdir(parents=True, exist_ok=True)
if not DATA.is_dir():
    raise FileNotFoundError("Extract the companion data ZIP and set OPAL2_DATA_ROOT to its zenodo_data directory.")
