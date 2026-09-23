"""Identity-level reservations for the next multi-dataset experiment.

This is an explicit check for future data loaders, not an operating-system
filesystem barrier. It does not read measurements or grant study approval.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


class IdentityReservationError(ValueError):
    pass


class IdentityReservations:
    def __init__(self, manifest_path):
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text())
        assignment_path = path.parent / self.manifest['identity_assignments']
        with assignment_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        self.identities = {}
        self.protected_groups = set(self.manifest['reserved_connectivity_groups'])
        for row in rows:
            key = (row['dataset'], row['object_id'])
            if key in self.identities:
                raise IdentityReservationError(f'Duplicate identity {key}')
            self.identities[key] = row

    def check(self, dataset, object_id, purpose, *, connectivity=None):
        """Return identity row or reject an unlisted / inappropriate access.

        Metadata are allowed for listed objects. All measurement access is
        closed in the current reservation version. A later release requires a
        separate manifest with explicit per-purpose role permissions.
        """
        row = self.identities.get((dataset, object_id))
        if row is None:
            raise IdentityReservationError('Identity is not in the declared manifest')
        if connectivity and row['connectivity'] and connectivity != row['connectivity']:
            raise IdentityReservationError('Object ID and chemical identity disagree')
        if purpose == 'metadata':
            return row
        if purpose not in {'model_fit', 'reference_fit', 'distribution_calibration',
                           'representation_pretraining', 'policy_calibration',
                           'evaluation_x', 'evaluation_outcomes'}:
            raise IdentityReservationError('Unknown access purpose')
        if row['identity_role'] in {'EU_RESERVED_HISTORICAL_OVERLAP_INELIGIBLE',
                                    'EU_RESERVED_PENDING_REVIEW'}:
            raise IdentityReservationError('Reserved identity is not eligible for measurement release')
        is_protected = (row['connectivity'] in self.protected_groups or
                        row.get('reserved_for_EU') in (True, 'True'))
        if is_protected and purpose not in {'evaluation_x', 'evaluation_outcomes'}:
            raise IdentityReservationError('Reserved evaluation identity cannot be a training/reference object')
        if not self.manifest['measurement_access_released']:
            raise IdentityReservationError('This manifest releases metadata only')
        grants = self.manifest.get('released_role_purposes', {}).get(row['identity_role'], [])
        if purpose not in grants:
            raise IdentityReservationError('Role is not released for this purpose')
        return row
