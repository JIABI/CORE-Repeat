"""RxRx3 approved-export adapter and negative-control-defined assay coordinates.

The raw source archive is not accepted here. Only the prior approved export is
read. Controls define the measurement space before any drug-outcome scoring;
the existing CORE subsequently fits its input/target transforms within TRAIN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from .biology_kernel_evaluation import write_json
from .eu_development_plan import allocate_groups


ROLE_KEYS = ('TRAIN', 'VALIDATION', 'REF_FIT', 'DIST_CAL', 'DEV_EVAL')
SEED = 20260918
MIN_DOSE_GROUPS = 500


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def fit_negative_control_space(values, plates, names):
    """Fixed EU-style assay recipe, applied to author-declared EMPTY controls.

    The 951 released columns have already been schema-checked as named CP
    measurements. Do not apply EU-specific compartment-prefix name filtering.
    All quantities below use controls only, never compound profiles.
    """
    values = np.asarray(values, np.float64)
    plates, names = np.asarray(plates, str), np.asarray(names, str)
    if values.shape != (len(plates), len(names)) or len(set(names)) != len(names):
        raise ValueError('Misaligned control schema')
    keep = np.ones(len(names), bool)
    centers, scales, counts = {}, {}, {}
    for plate in np.unique(plates):
        block = values[plates == plate]
        if len(block) < 8:
            raise ValueError('Fewer than eight authorized negative controls on '+plate)
        finite = np.isfinite(block).all(0)
        center, scale = np.full(len(names), np.nan), np.full(len(names), np.nan)
        center[finite] = np.median(block[:, finite], axis=0)
        scale[finite] = np.median(np.abs(block[:, finite]-center[finite]), axis=0)
        keep &= finite & np.isfinite(scale) & (scale > 1e-12)
        centers[plate], scales[plate], counts[plate] = center, scale, len(block)
    if keep.sum() < 8:
        raise ValueError('Fewer than eight common control-valid features')
    return dict(source_feature_names=names.tolist(), feature_names=names[keep].tolist(),
        selected_indices=np.flatnonzero(keep).tolist(),
        centers={key: value[keep].tolist() for key, value in centers.items()},
        scales={key: value[keep].tolist() for key, value in scales.items()},
        control_counts=counts, control_kind='author-declared EMPTY_control, not relabelled DMSO',
        control_only=True, mad_multiplier=1., clip=10., nonfinite_fill=0.,
        maximum_nonfinite_fraction=.05,
        feature_rule='released named CP features; finite controls and unscaled MAD >1e-12 on every authorized role plate')


def apply_negative_control_space(y, plates, space):
    y, plates = np.asarray(y, np.float64), np.asarray(plates, str)
    if y.ndim != 3 or y.shape[:2] != plates.shape or y.shape[1] != 4:
        raise ValueError('Four roles and physical plate identities must align')
    if y.shape[2] != len(space['source_feature_names']):
        raise ValueError('Source measurement feature schema changed')
    indices = np.asarray(space['selected_indices'], int)
    flat, flat_plates = y.reshape(-1, y.shape[-1]), plates.ravel()
    out = np.empty((len(flat), len(indices)), np.float64)
    if set(flat_plates)-set(space['centers']):
        raise ValueError('A role plate lacks authorized controls')
    for plate in np.unique(flat_plates):
        rows = np.flatnonzero(flat_plates == plate)
        out[rows] = (flat[np.ix_(rows, indices)]-np.asarray(space['centers'][plate]))/np.asarray(space['scales'][plate])
    finite = np.isfinite(out)
    fractions = (~finite).mean(1)
    filled = np.clip(np.where(finite, out, space['nonfinite_fill']), -space['clip'], space['clip'])
    bad = (fractions > space['maximum_nonfinite_fraction']) | (np.linalg.norm(filled, axis=1) == 0)
    if bad.any():
        raise ValueError(f'{int(bad.sum())} approved role wells fail the predeclared technical validity rule; no automatic object deletion')
    audit = dict(n_conditions=len(y), n_role_wells=len(flat), retained_features=len(indices),
        nonfinite_coordinates_filled=int((~finite).sum()),
        clipped_coordinates=int((finite & (np.abs(out) > space['clip'])).sum()),
        all_drug_identities_retained=True, drug_outcomes_used_to_fit_transform=False)
    return filled.reshape(len(y), 4, -1), audit


def chemistry(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    out = np.zeros((len(smiles), 513), np.float64)
    cache = {}
    for i, text in enumerate(smiles):
        text = str(text)
        if text not in cache:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                raise ValueError('An approved identity has no valid SMILES; no silent scope change')
            bits = np.zeros(512, np.float64)
            DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
            cache[text] = bits
        out[i, :512], out[i, 512] = cache[text], 1.
    return out, np.ones(len(smiles), bool)


def eligible_doses(ids, groups, dose, minimum_groups=MIN_DOSE_GROUPS):
    """Metadata-only sufficiency for a complete, unchanged per-dose CORE.

    500 chemical groups supply approximately 80 REF groups under the fixed
    5-fold/60:20:20 split, above the 64-anchor requirement. The actual distinct
    fingerprint requirement is verified again before training. No dose label
    is rounded or merged, and excluded conditions remain in the approved export.
    """
    ids, groups, dose = np.asarray(ids, str), np.asarray(groups, str), np.asarray(dose, float)
    keep = np.zeros(len(ids), bool)
    excluded = []
    for level in np.unique(dose):
        rows = dose == level
        n_groups = len(set(groups[rows]))
        if n_groups >= minimum_groups:
            keep |= rows
        else:
            excluded.append(dict(dose_uM=float(level), n_conditions=int(rows.sum()),
                n_chemical_groups=n_groups, condition_ids=ids[rows].tolist(),
                reason='metadata-only insufficient support for complete dose-specific fitting/reference/calibration'))
    return keep, excluded


def build_dataset(export, output):
    export, output = Path(export), Path(output)
    if (output/'complete.json').exists():
        return read_npz(output/'data.npz'), json.loads((output/'metadata.json').read_text())
    raw = read_npz(export/'measurements.npz')
    control = read_npz(export/'controls.npz')
    if len(set(raw['ids'])) != len(raw['ids']) or set(raw['well_ids'].ravel()) & set(control['well_ids']):
        raise ValueError('Approved drug and control identities overlap or condition IDs duplicate')
    np.testing.assert_array_equal(raw['feature_names'], control['feature_names'])
    # The export fixes this physical resource before numerical assay fitting.
    if set(raw['plates'].ravel()) != set(control['plates']):
        raise ValueError('Authorized controls must cover exactly the declared role plates')
    approved_n = len(raw['ids'])
    eligible, excluded = eligible_doses(raw['ids'], raw['groups'], raw['dose'])
    if not eligible.any():
        raise ValueError('No dose supports the unchanged complete R2 pipeline')
    raw = {key: value[eligible] if key not in {'feature_names', 'roles'} and
        value.ndim > 0 and value.shape[0] == approved_n else value for key, value in raw.items()}
    space = fit_negative_control_space(control['Y'], control['plates'], raw['feature_names'])
    y, audit = apply_negative_control_space(raw['Y'], raw['plates'], space)
    chem, mask = chemistry(raw['smiles'])
    data = {key: raw[key] for key in ('ids', 'groups', 'object_ids', 'well_ids', 'plates', 'dose', 'batches')}
    data['dose'] = np.asarray(data['dose'], float)
    # Experiment is a conservative shared-plate clustering unit. A random role
    # permutation must not create many artificial independent layout blocks.
    batch = np.asarray(raw['batches'], str)
    if batch.ndim == 2:
        if np.any(batch != batch[:, :1]):
            raise ValueError('Four role wells must share one experiment')
        batch = batch[:, 0]
    data.update(Y=y, chem=chem, chem_mask=mask, layout=batch,
        feature_names=np.asarray(space['feature_names']))
    metadata = dict(dataset='RxRx3-core', scope='approved chemical MODULE_DEV only',
        n_conditions=len(y), n_chemical_groups=len(set(data['groups'])),
        approved_export_conditions=approved_n, metadata_only_unsupported_doses=excluded,
        minimum_dose_chemical_groups=MIN_DOSE_GROUPS,
        doses_uM=sorted(np.unique(data['dose']).astype(float).tolist()),
        morphology_dimension=y.shape[-1], source_feature_dimension=raw['Y'].shape[-1],
        chemical=dict(kind='Morgan binary fingerprint', radius=2, bits=512,
            final_coordinate='valid_SMILES_indicator', rdkit_version=rdBase.rdkitVersion,
            grouping='preassigned standardized connectivity, all doses together'),
        measurement_space='authorized EMPTY controls only: per-plate median/unscaled MAD; common positive-finite-MAD features; clip +/-10',
        control_wells=len(control['Y']), assay_space_fitted_from_compound_outcomes=False,
        model_preprocessing='full CORE TRAIN-only input/target affine transforms',
        control_availability='existing same-experiment assay controls; purchased setup costs reported separately',
        source_preprocessing_provenance='named released CP features; any release-time scaling not fully documented',
        dose_is_model_stratum=True, endpoint='same-condition ADD_TWO half-cosine gain minus 0.02',
        biology_active=False, representation_active=False, confirmation_opened=False)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'data.npz', **data)
    write_json(output/'metadata.json', metadata)
    write_json(output/'control_space.json', space)
    write_json(output/'assay_audit.json', audit)
    write_json(output/'complete.json', dict(complete=True, source_export=str(export),
        compound_outcomes_used_to_choose_preprocessing=False))
    return data, metadata


def grouped_parts(data, seed=SEED):
    """One chemical allocation shared by every dose; no measurement input."""
    ids, groups = np.asarray(data['ids'], str), np.asarray(data['groups'], str)
    dose, layout = np.asarray(data['dose'], float), np.asarray(data['layout'], str)
    if len(set(ids)) != len(ids):
        raise ValueError('Condition IDs must be unique')
    strata = {group: '|'.join(sorted(set(layout[groups == group]))) for group in np.unique(groups)}
    outer = allocate_groups(strata, list(range(5)), [1.]*5, seed)
    outer_fold = np.asarray([outer[group] for group in groups], int)
    parts, records = [], []
    for f in range(5):
        available = {group: value for group, value in strata.items() if outer[group] != f}
        phase = allocate_groups(available, ['MODEL_FIT', 'REF_FIT', 'DIST_CAL'], [.6, .2, .2], seed+100+f)
        model = allocate_groups({g: s for g, s in available.items() if phase[g] == 'MODEL_FIT'},
            ['TRAIN', 'VALIDATION'], [.8, .2], seed+1000+f)
        role = np.asarray(['DEV_EVAL' if outer[g] == f else model.get(g, phase[g]) for g in groups])
        for level in np.unique(dose):
            part = {key: np.flatnonzero((dose == level) & (role == key)) for key in ROLE_KEYS}
            if any(not len(rows) for rows in part.values()):
                raise ValueError('Empty dose-specific fitting/reference/evaluation role')
            for i, key in enumerate(ROLE_KEYS):
                for other in ROLE_KEYS[:i]:
                    if set(groups[part[key]]) & set(groups[part[other]]):
                        raise ValueError('A chemical group crosses roles')
            records.append(dict(cell=len(parts), outer_fold=f, dose_uM=float(level),
                counts={key: len(rows) for key, rows in part.items()},
                group_counts={key: len(set(groups[rows])) for key, rows in part.items()}))
            parts.append(part)
    count = np.zeros(len(ids), int)
    for part in parts:
        count[part['DEV_EVAL']] += 1
    if not np.all(count == 1):
        raise ValueError('Every approved condition requires exactly one OOF prediction')
    return parts, records, outer_fold
