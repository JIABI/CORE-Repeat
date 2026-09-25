"""Render all current main and supplementary figures from saved observations."""
import argparse
from pathlib import Path
import subprocess
import sys

COMMANDS = {
    "1": ["plot_measurement_fig1.py"],
    "2": ["plot_data_rich_results.py", "4"],
    "3": ["plot_data_rich_results.py", "2"],
    "4": ["plot_m3_m4_controls.py"],
    "5": ["plot_m4_measurement_coverage.py"],
    "6": ["plot_data_rich_results.py", "5"],
    "7": ["plot_data_rich_fig3.py"],
    "S3": ["plot_figures.py"],
    "S5": ["plot_layout_diagnostics_20260923.py"],
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("figures", nargs="*", choices=list(COMMANDS))
    args = p.parse_args()
    scripts = Path(__file__).resolve().parent / "scripts"
    for name in args.figures or COMMANDS:
        cmd = COMMANDS[name]
        print("Rendering", name, flush=True)
        subprocess.run([sys.executable, str(scripts / cmd[0]), *cmd[1:]], check=True)
