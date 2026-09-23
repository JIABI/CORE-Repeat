"""Control-defined EU measurement space, separate from model preprocessing.

Accepts only a prior FIT/control export. No download or confirmation-data API.
All arms use the same measured coordinates; learned input/target scaling stays
inside MODEL_FIT in the complete CORE trainer.
"""
from __future__ import annotations

import numpy as np


# EU aggregated files use Cyto_/Nuc_; the long aliases occur in other CP exports.
COMPARTMENTS = ('Cells_', 'Cyto_', 'Nuc_', 'Cytoplasm_', 'Nuclei_')


def measurement_columns(names):
    names = np.asarray(names, str)
    if names.ndim != 1 or len(set(names)) != len(names):
        raise ValueError('Unique one-dimensional feature names required')
    return np.array([name.startswith(COMPARTMENTS)
        and not name.endswith('_Number_Object_Number')
        and '_Parent_' not in name and 'ClosestObjectNumber_' not in name
        for name in names], dtype=bool)


def fit_control_space(values, plate_ids, is_dmso, feature_names):
    """Median/MAD from declared DMSO only; keep common usable coordinates.

    MAD is the unscaled median absolute deviation. The 1.4826 convention is
    deliberately not implicit. Require finite DMSO values and MAD > 1e-12 in
    every declared plate; do not set a tiny denominator or inspect drug outcomes
    to select features. This defines the new EU assay space, not an alteration
    of a previously measured LINCS endpoint.
    """
    values = np.asarray(values, np.float64)
    plates, dmso, names = np.asarray(plate_ids, str), np.asarray(is_dmso), np.asarray(feature_names, str)
    if (values.ndim != 2 or values.shape != (len(plates), len(names))
            or dmso.shape != plates.shape or dmso.dtype != bool):
        raise ValueError('Aligned values, plates, boolean DMSO mask and schema required')
    eligible = measurement_columns(names)
    location, scale, audits = {}, {}, []
    for plate in sorted(set(plates)):
        controls = values[(plates == plate) & dmso]
        if len(controls) < 8:
            raise ValueError('Insufficient declared DMSO controls on plate '+plate)
        finite = np.isfinite(controls).all(0)
        center, mad = np.full(len(names), np.nan), np.full(len(names), np.nan)
        center[finite] = np.median(controls[:, finite], axis=0)
        mad[finite] = np.median(np.abs(controls[:, finite]-center[finite]), axis=0)
        usable = finite & np.isfinite(mad) & (mad > 1e-12)
        eligible &= usable
        location[plate], scale[plate] = center, mad
        audits.append(dict(plate=plate, n_dmso=len(controls), finite_control_features=int(finite.sum()),
                           positive_mad_features=int(usable.sum())))
    if eligible.sum() < 8:
        raise ValueError('Fewer than eight common control-valid morphology coordinates')
    return dict(source_feature_names=names.tolist(), feature_names=names[eligible].tolist(),
        selected_indices=np.flatnonzero(eligible).tolist(),
        centers={p:v[eligible].tolist() for p,v in location.items()},
        scales={p:v[eligible].tolist() for p,v in scale.items()},
        plates=audits, control_only=True, mad_multiplier=1.0,
        clip=10.0, nonfinite_fill=0.0, maximum_nonfinite_fraction=0.05,
        feature_rule='CP compartment measurements excluding object/parent identifiers; '
            'all DMSO finite and unscaled MAD >1e-12 on every declared plate')


def apply_control_space(values, plate_ids, space, *, required_wells=None):
    values, plates = np.asarray(values, np.float64), np.asarray(plate_ids, str)
    if values.ndim != 2 or values.shape != (len(plates), len(space['source_feature_names'])):
        raise ValueError('Measurement schema differs from fitted control space')
    required = np.ones(len(plates), bool) if required_wells is None else np.asarray(required_wells)
    if required.shape != plates.shape or required.dtype != bool:
        raise ValueError('Required-well mask must be boolean and aligned')
    if set(plates)-set(space['centers']):
        raise ValueError('Plate has no declared control transform')
    ix = np.asarray(space['selected_indices'], int)
    standardized = np.empty((len(plates), len(ix)), np.float64)
    for plate in sorted(set(plates)):
        rows = plates == plate
        standardized[rows] = ((values[rows][:, ix]-np.asarray(space['centers'][plate]))
                               /np.asarray(space['scales'][plate]))
    finite = np.isfinite(standardized)
    fraction = (~finite).mean(1)
    normalized = np.clip(np.where(finite, standardized, space['nonfinite_fill']),
                         -space['clip'], space['clip'])
    failed = required & ((fraction > space['maximum_nonfinite_fraction'])
                         | (np.linalg.norm(normalized, axis=1) == 0))
    if failed.any():
        # Row indices only: do not print profile values and do not silently drop.
        raise ValueError('Required FIT/control export has technically invalid drug wells at rows '
                         +str(np.flatnonzero(failed).tolist()))
    report = dict(n_wells=len(plates), n_features=len(ix),
        nonfinite_filled_count=(~finite).sum(1).tolist(),
        clipped_coordinate_count=(finite & (np.abs(standardized)>space['clip'])).sum(1).tolist(),
        objects_removed=False, learned_from_drug_outcomes=False)
    return normalized, report
