"""EU external biological annotations and full reference-error summaries.

The two relations are reported target identity and (target, reported action).
An ``inhibitor`` action without its target is not an MoA identity. Annotation
resources contain external prior knowledge, never EU screening outcomes.
"""
from __future__ import annotations

import csv
import json
import posixpath
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np

from .biology_random_reference_experiment import matched_random_weights
from .dual_branch_features import FIELDS, NAMES, biology_features
from .joint_contrast_scale import contrast_projector
from .module_switch_experiment import biology_similarity, context_mask


PROJECT = Path(__file__).resolve().parents[1]
MISSING = {None, "", "-", "NA", "N/A", "None", "null"}


def _text(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value in MISSING else value


def _human(value):
    if value is True or str(value).strip().lower() in {"1", "true"}:
        return True
    if value is False or str(value).strip().lower() in {"0", "false"}:
        return False
    return None


def _rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _xlsx_rows(archive, sheet_name):
    """Read cached scalar cells from the source OOXML annotation workbook.

    This is metadata ingestion, with no optional spreadsheet-library dependency
    in the experiment runtime. Formula cells, if present, use cached values.
    """
    main = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    shared = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        tree = ET.fromstring(archive.read('xl/sharedStrings.xml'))
        shared = [''.join(t.text or '' for t in cell.iter('{'+main+'}t')) for cell in tree]
    book = ET.fromstring(archive.read('xl/workbook.xml'))
    sheets = book.find('{'+main+'}sheets')
    selected = [s for s in sheets if s.attrib.get('name') == sheet_name]
    if len(selected) != 1:
        raise ValueError('Missing or ambiguous annotation worksheet: '+sheet_name)
    rid = selected[0].attrib['{'+rel+'}id']
    links = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    link = [v for v in links if v.attrib.get('Id') == rid]
    if len(link) != 1 or link[0].attrib.get('TargetMode') == 'External':
        raise ValueError('Invalid annotation worksheet relationship')
    target = link[0].attrib['Target']
    path = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
    tree = ET.fromstring(archive.read(path))
    for row in tree.find('{'+main+'}sheetData'):
        values = []
        for cell in row:
            address = cell.attrib['r']
            letters = ''.join(v for v in address if v.isalpha())
            column = 0
            for letter in letters:
                column = column*26+ord(letter.upper())-ord('A')+1
            while len(values) < column:
                values.append(None)
            kind = cell.attrib.get('t', 'n')
            element = cell.find('{'+main+'}v')
            value = element.text if element is not None else None
            if kind == 'inlineStr':
                value = ''.join(t.text or '' for t in cell.iter('{'+main+'}t'))
            elif kind == 's' and value is not None:
                value = shared[int(value)]
            elif kind == 'b' and value is not None:
                value = value == '1'
            elif kind == 'n' and value is not None:
                number = float(value)
                value = int(number) if number.is_integer() else number
            elif kind == 'e':
                raise ValueError('Source annotation contains an Excel error')
            values[column-1] = value
        yield values


def _incidence(values, vocabulary):
    index = {name: j for j, name in enumerate(vocabulary)}
    matrix = np.zeros((len(values), len(index)), float)
    for i, names in enumerate(values):
        for name in names:
            matrix[i, index[name]] = 1.
    return matrix, np.asarray([bool(v) for v in values], bool)


def load_eu_biology_metadata(ids, groups, *, root=PROJECT, human_only=True,
                            identity_csv=None, annotation_workbook=None,
                            well_metadata_csv=None, phase_manifest=None):
    """Load annotations for explicitly allowlisted development identities only.

    No profile, cell-count, image, embedding or outcome file is opened. All
    compound/target rows retain their original action, human flag and sources.
    Unknown or ineligible relations have a false availability mask, not a claim
    that the compound has no biological target. Defaults resolve existing files.
    """
    root = Path(root)
    qualification = root/'reports/new_data_qualification_20260917_v1/eu_openscreen'
    phase = root/'reports/eu_core_development_20260917_v1'
    identity_csv = Path(identity_csv or qualification/'identity_raw.csv')
    annotation_workbook = Path(annotation_workbook or qualification/'source_metadata/Suppl_Table_9.xlsx')
    well_metadata_csv = Path(well_metadata_csv or phase/'ingest/row_metadata.csv')
    phase_manifest = Path(phase_manifest or phase/'phase_manifest.json')
    ids, groups = np.asarray(ids, str), np.asarray(groups, str)
    if ids.ndim != 1 or len(ids) == 0 or groups.shape != ids.shape or len(set(ids)) != len(ids):
        raise ValueError('Unique aligned development IDs and chemical groups required')
    manifest = json.loads(phase_manifest.read_text())
    if not set(ids) <= set(manifest['allowed_compound_ids']):
        raise ValueError('Annotation request includes identities outside the development allowlist')
    identity = {r['object_id']: r for r in _rows(identity_csv) if r['object_id'] in set(ids)}
    if set(identity) != set(ids) or len({r['pdid'] for r in identity.values()}) != len(ids):
        raise ValueError('Development identity to PDID mapping is incomplete or ambiguous')
    by_pdid = {r['pdid']: oid for oid, r in identity.items()}
    workbook = ZipFile(annotation_workbook)
    try:
        iterator = _xlsx_rows(workbook, 'COMPOUNDS')
        header = next(iterator)
        compounds = {}
        for row in iterator:
            value = dict(zip(header, row))
            if value['pdid'] in by_pdid:
                if value['pdid'] in compounds:
                    raise ValueError('Duplicate annotated PDID')
                compounds[value['pdid']] = value
        if set(compounds) != set(by_pdid):
            raise ValueError('Missing annotated compound identity')
        for pdid, oid in by_pdid.items():
            for field in ('smiles', 'inchi', 'inchikey'):
                if str(compounds[pdid].get(field)) != identity[oid][field]:
                    raise ValueError(f'Annotation identity mismatch: {oid}/{field}')
        iterator = _xlsx_rows(workbook, 'TARGETS')
        header = next(iterator)
        relations = []
        targets = {oid: set() for oid in ids}
        actions = {oid: set() for oid in ids}
        raw_target, raw_action = set(), set()
        human_counts = {'human': 0, 'nonhuman': 0, 'unknown': 0}
        for excel_row, row in enumerate(iterator, start=2):
            value = dict(zip(header, row))
            if value['pdid'] not in by_pdid:
                continue
            oid = by_pdid[value['pdid']]
            target, action = _text(value.get('target_name')), _text(value.get('moa'))
            human = _human(value.get('human'))
            human_counts['human' if human is True else 'nonhuman' if human is False else 'unknown'] += 1
            eligible = target is not None and (not human_only or human is True)
            if target is not None:
                raw_target.add(oid)
            if target is not None and action is not None:
                raw_action.add(oid)
            # Keep composites and conflicting/multiple reported actions intact.
            # No inferred gene splitting, direction, potency or EC50 conversion.
            if eligible:
                targets[oid].add(target)
                if action is not None:
                    actions[oid].add(json.dumps([target, action], ensure_ascii=False))
            relations.append(dict(object_id=oid, source_row=excel_row, raw=value,
                                  human_status=human, eligible_target=eligible,
                                  eligible_target_action=eligible and action is not None))
    finally:
        workbook.close()
    target_names = sorted(set().union(*targets.values()))
    moa_names = sorted(set().union(*actions.values()))
    target, target_mask = _incidence([targets[i] for i in ids], target_names)
    moa, moa_mask = _incidence([actions[i] for i in ids], moa_names)
    wells = {oid: {} for oid in ids}
    for row in _rows(well_metadata_csv):
        oid = row.get('object_id')
        if oid not in wells or row.get('resource_kind') != 'FIT_COMPOUND':
            continue
        role = row['measurement_role']
        if role not in ('X', 'Z1', 'Z2', 'V') or role in wells[oid]:
            raise ValueError('Ambiguous physical well role')
        wells[oid][role] = row
    units = []
    for oid, group in zip(ids, groups):
        rows = wells[oid]
        if set(rows) != {'X', 'Z1', 'Z2', 'V'}:
            raise ValueError('Development metadata lacks four physical roles: '+oid)
        if {r['connectivity'] for r in rows.values()} != {group}:
            raise ValueError('Chemical grouping differs from the frozen role metadata')
        conditions = {(r['site'], r['cell'], r['dose_record'], r['dose_unit'],
                       r['exposure_h_protocol']) for r in rows.values()}
        if len(conditions) != 1:
            raise ValueError('Four-role condition mismatch')
        site, cell, dose, unit, time = next(iter(conditions))
        if unit not in {'uM', 'µM', 'μM'} or not cell or not site:
            raise ValueError('Missing dose unit/cell/site condition')
        dose, time = float(dose), float(time)
        if not np.isfinite([dose, time]).all() or dose <= 0 or time <= 0:
            raise ValueError('Invalid declared dose or exposure')
        units.append(dict(id=oid, compound_id=oid, cell_line=cell, site=site,
            platform='EU_OPENSCREEN:cpg0036:CellPainting',
            actual_dose_uM=dose, dose_evidence='plate-map value and published unit',
            exposure_hours_protocol_nominal=time, exact_execution_verified=False,
            layout_block=rows['X']['library_plate'],
            roles={role: dict(plate=r['plate_uid'], well=r['well_position'],
                             well_id=r['well_id'], cell_count=None)
                   for role, r in rows.items()}))
    arrays = dict(target=target, target_mask=target_mask, moa=moa, moa_mask=moa_mask)
    report = dict(n=len(ids), human_only=bool(human_only), raw_target_coverage=len(raw_target),
        raw_target_action_coverage=len(raw_action), eligible_target_coverage=int(target_mask.sum()),
        eligible_target_action_coverage=int(moa_mask.sum()), eligible_target_vocabulary=len(target_names),
        eligible_target_action_vocabulary=len(moa_names), relation_rows=len(relations),
        human_relation_counts=human_counts, target_semantics='Exact reported target_name; composite targets retained',
        moa_semantics='Exact (target_name, reported moa/action string) pair; not action alone',
        unknown_relation_semantics='mask false means unavailable/ineligible, never no biological target',
        vocabulary_scope='External annotations of the supplied development IDs; no outcome-derived vocabulary',
        context_evidence='Same site/cell/platform, plate-map dose, nominal protocol time; execution not verified',
        source=str(annotation_workbook), identity_source=str(identity_csv),
        well_metadata_source=str(well_metadata_csv), phase_manifest=str(phase_manifest),
        source_version='Zenodo19347244 v3 Supplementary Table 9, external Probes & Drugs priors',
        measurements_read=False, biological_activity_values_used=False)
    return dict(arrays=arrays, metadata=dict(units=units, target_names=target_names, moa_names=moa_names),
                relations=relations, report=report)


def _permissions(data, metadata, query, donors, allowed):
    n, m = len(query), len(donors)
    permit = np.ones((n, m), bool) if allowed is None else np.asarray(allowed)
    if permit.dtype != bool or permit.shape != (n, m):
        raise ValueError('Reference permissions must be an aligned Boolean matrix')
    permit = permit.copy() & context_mask(metadata, query, donors)
    extra_keys = () if metadata.get('reference_context_policy') == 'rxrx3_protocol_range_v1' else ('site', 'platform')
    for key in extra_keys:
        q = [metadata['units'][i].get(key) for i in query]
        d = [metadata['units'][i].get(key) for i in donors]
        permit &= np.asarray([[a is not None and b is not None and a == b for b in d] for a in q])
    permit &= data['groups'][query, None] != data['groups'][None, donors]
    return permit


def _random_summaries(data, query, donors, raw_mean, covariance, residual,
                      permit, relations, original_mass):
    """Same complete 24-field summary, with randomized reference assignments.

    Only the donor identities carrying the original similarity weights differ
    from dual_branch_features.biology_features. Every donor error is whitened
    in the current query's common native geometry frame.
    """
    decomposition = contrast_projector(raw_mean, np.ones(9), covariance)
    x = np.asarray(data['Y'][:, 0], float)
    norm = np.linalg.norm(x, axis=1)
    direction = x / norm[:, None]
    cosine = np.clip(direction[query] @ direction[donors].T, -1., 1.)
    amp_gap = np.abs(np.log(norm[query, None])-np.log(norm[None, donors]))
    out = np.zeros((len(query), len(NAMES)))
    support = np.zeros((len(query), 2), bool)
    for i in range(len(query)):
        whitened = np.linalg.solve(decomposition['factor'][i], residual.T).T
        pair = whitened @ decomposition['projector'][i]
        energy = np.column_stack((np.square(pair).sum(1)/3., np.square(whitened-pair).sum(1)/6.))
        pool = energy[permit[i]]
        pool_variance = np.var(pool, axis=0, ddof=1) if len(pool) > 1 else np.array([2/3, 2/6])
        pool_variance = np.maximum(pool_variance, [2/3, 2/6])
        for r, similarity in enumerate(relations):
            s = similarity[i]
            count, mass = int((s > 0).sum()), float(original_mass[r][i])
            if not count:
                continue
            weights = s / mass
            ess = 1 / float(np.square(weights).sum())
            mean = weights @ energy
            denominator = 1-float(np.square(weights).sum())
            variance = weights @ np.square(energy-mean)/denominator if denominator > 1e-12 else pool_variance
            se = np.sqrt(np.maximum(variance, 0)/ess)
            confidence = ess/(ess+5.)*(-np.expm1(-mass))
            out[i, r*len(FIELDS):(r+1)*len(FIELDS)] = [
                1., np.log1p(count), np.log1p(mass), np.log1p(ess), mass/count,
                *np.log1p(mean), *np.log1p(se), float(weights @ amp_gap[i]),
                float(weights @ (1-cosine[i])), confidence]
            support[i, r] = True
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite randomized biological summaries')
    return dict(values=out, names=NAMES.copy(), support=support.any(1), support_by_relation=support)


def build_eu_biology_features(data, metadata, query, donors, raw_mean, raw_covariance,
                              donor_raw_residual, *, allowed=None, random_seed=None,
                              amplitude_edges=None):
    """Full biological summaries, or count/weight/condition-matched random ones.

    The caller supplies honest donor residuals and query mean/covariance, all
    in the same native nine-dimensional coordinate system. Optional amplitude
    edges must be fitted using training X only. Randomization uses stable query
    and donor identities, never future query measurements or donor outcomes.
    """
    query, donors = np.asarray(query, int), np.asarray(donors, int)
    ids = np.asarray(data['ids'], str)
    if (query.ndim != 1 or donors.ndim != 1 or len(set(query)) != len(query)
            or len(set(donors)) != len(donors) or len(set(ids)) != len(ids)
            or np.any(np.r_[query, donors] < 0) or np.any(np.r_[query, donors] >= len(ids))):
        raise ValueError('Unique aligned query and donor indices required')
    permit = _permissions(data, metadata, query, donors, allowed)
    # Real BIO is exactly the full existing implementation, not an approximation.
    original = biology_features(data, metadata, query, donors, raw_mean, raw_covariance,
                                donor_raw_residual, allowed=permit)
    original['audit'] = dict(randomized=False, native_query_whitening=True,
        relation_semantics=metadata.get('moa_semantics', 'target plus reported action'),
        query_ids=ids[query].tolist(), donor_ids=ids[donors].tolist(), legal_counts=permit.sum(1))
    if random_seed is None:
        return original
    if isinstance(random_seed, bool) or int(random_seed) != random_seed or random_seed < 0:
        raise ValueError('Nonnegative integer randomization seed required')
    amp = np.log(np.linalg.norm(np.asarray(data['Y'][:, 0], float), axis=1))
    if not np.isfinite(amp[np.r_[query, donors]]).all():
        raise ValueError('Finite first-well amplitude required')
    edges = np.array([], float) if amplitude_edges is None else np.asarray(amplitude_edges, float)
    if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(np.diff(edges) < 0):
        raise ValueError('Amplitude stratum boundaries must be finite and ordered')
    bins = np.searchsorted(edges, amp[donors], side='right')
    source = [s*permit for s in biology_similarity(data, metadata, query, donors)]
    mass = [s.sum(1) for s in source]
    randomized, diagnostics = [], []
    order = np.argsort(ids[donors], kind='stable')
    for relation, similarities in enumerate(source):
        shuffled = np.zeros_like(similarities)
        maps = np.full(similarities.shape, -1, int)
        fixed_mass, retained_mass, displacement = [], [], []
        for i, qid in enumerate(ids[query]):
            normalized = similarities[i:i+1]/mass[relation][i] if mass[relation][i] > 0 else similarities[i:i+1]
            seed = np.random.SeedSequence([int(random_seed), relation, *qid.encode('utf-8')])
            _, audit = matched_random_weights(normalized[:, order], permit[i:i+1, order],
                bins[order], amp[donors][order], seed=seed)
            positions = np.flatnonzero(similarities[i, order] > 0)
            destinations = audit['donor_mapping'][0, positions]
            # Copy original values rather than renormalizing, retaining their
            # exact positive multiset and their original total relation strength.
            shuffled[i, order[destinations]] = similarities[i, order[positions]]
            maps[i, order[positions]] = order[destinations]
            fixed_mass.append(float(audit['fixed_mass'][0]))
            retained_mass.append(float(audit['retained_weight_mass'][0]))
            displacement.append(float(audit['weighted_log_amplitude_displacement'][0]))
        np.testing.assert_array_equal(np.sort(shuffled, axis=1), np.sort(similarities, axis=1))
        np.testing.assert_array_equal(shuffled[~permit], np.zeros(np.count_nonzero(~permit)))
        randomized.append(shuffled)
        diagnostics.append(dict(relation=('target', 'moa')[relation], donor_mapping=maps,
            original_similarity=similarities, randomized_similarity=shuffled,
            positive_count=(similarities > 0).sum(1), fixed_mass=np.asarray(fixed_mass),
            retained_weight_mass=np.asarray(retained_mass),
            weighted_log_amplitude_displacement=np.asarray(displacement)))
    result = _random_summaries(data, query, donors, np.asarray(raw_mean, float),
        np.asarray(raw_covariance, float), np.asarray(donor_raw_residual, float), permit, randomized, mass)
    # Support quantities must be identical, not rounded variations induced by
    # floating-point addition order. Only reference error/condition summaries vary.
    invariant = [r*len(FIELDS)+j for r in range(2) for j in (0, 1, 2, 3, 4, 11)]
    result['values'][:, invariant] = original['values'][:, invariant]
    np.testing.assert_array_equal(result['support_by_relation'], original['support_by_relation'])
    result['audit'] = dict(original['audit'], randomized=True, seed=int(random_seed),
        amplitude_stratified=amplitude_edges is not None, amplitude_edges=edges,
        relation_diagnostics=diagnostics, positive_weights_preserved=True,
        positive_count_and_ess_preserved=True, outcome_used_for_matching=False)
    return result
