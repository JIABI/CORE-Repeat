"""Build the approved EU FIT-only development metadata plan, without measurements."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from opal2.eu_development_plan import read_csv, write_plan


def main():
    assignment = ROOT / 'reports/new_data_assignment_20260917_v1'
    qualification = ROOT / 'reports/new_data_qualification_20260917_v1/eu_openscreen'
    output = ROOT / 'reports/eu_core_development_20260917_v1'
    summary = write_plan(output,
        read_csv(assignment / 'identity_assignments.csv'),
        read_csv(qualification / 'well_metadata.csv'),
        json.loads((assignment / 'reservation_manifest.json').read_text()))
    compact = {key: value for key, value in summary.items() if key != 'outer_folds'}
    compact['outer_folds'] = [{key: value for key, value in fold.items() if key != 'stratification_counts'}
                             for fold in summary['outer_folds']]
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
