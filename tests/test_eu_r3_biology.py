"""EU annotation semantics, complete feature reuse and matched-reference tests."""
from copy import deepcopy
import csv
import json
from zipfile import ZipFile
from xml.sax.saxutils import escape

import numpy as np
import pytest

from opal2.dual_branch_features import biology_features, FIELDS
from opal2.eu_r3_biology import load_eu_biology_metadata, build_eu_biology_features


def feature_inputs():
    rng = np.random.default_rng(1609)
    n = 13
    data = dict(ids=np.array([f'id_{i:02}' for i in range(n)]),
        groups=np.array([f'g{i}' for i in range(n)]), Y=rng.normal(size=(n, 4, 18)),
        target=np.eye(3)[[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0]],
        moa=np.eye(3)[[1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]],
        target_mask=np.ones(n, bool), moa_mask=np.ones(n, bool))
    metadata = dict(units=[dict(id=oid, site='FMP', platform='CP', cell_line='HepG2',
        exposure_hours_protocol_nominal=24., actual_dose_uM=10.) for oid in data['ids']])
    q, d = np.arange(3), np.arange(3, n)
    mean = rng.normal(size=(len(q), 9))*.1
    factor = np.eye(9)[None]+rng.normal(size=(len(q), 9, 9))*.02
    covariance = factor @ factor.transpose(0, 2, 1)
    residual = rng.normal(size=(len(d), 9))
    return data, metadata, q, d, mean, covariance, residual


def test_real_features_are_exact_full_existing_implementation():
    args = feature_inputs()
    out = build_eu_biology_features(*args)
    old = biology_features(*args)
    for key in ('values', 'support', 'support_by_relation'):
        np.testing.assert_array_equal(out[key], old[key])
    assert out['names'] == old['names'] and out['values'].shape == (3, 24)


def test_random_control_preserves_support_strength_multiset_and_amplitude_strata():
    args = feature_inputs()
    amp = np.log(np.linalg.norm(args[0]['Y'][:, 0], axis=1))
    edges = np.quantile(amp[args[3]], [.2, .4, .6, .8])
    out = build_eu_biology_features(*args, random_seed=741, amplitude_edges=edges)
    real = build_eu_biology_features(*args)
    np.testing.assert_array_equal(out['support'], real['support'])
    invariant = [r*len(FIELDS)+j for r in range(2) for j in (0, 1, 2, 3, 4, 11)]
    np.testing.assert_array_equal(out['values'][:, invariant], real['values'][:, invariant])
    bins = np.searchsorted(edges, amp[args[3]], side='right')
    for diag in out['audit']['relation_diagnostics']:
        a, b = diag['original_similarity'], diag['randomized_similarity']
        np.testing.assert_array_equal(np.sort(a, axis=1), np.sort(b, axis=1))
        for i in range(len(a)):
            src = np.flatnonzero(a[i])
            dest = diag['donor_mapping'][i, src]
            np.testing.assert_array_equal(bins[src], bins[dest])
    assert out['audit']['outcome_used_for_matching'] is False


def test_random_mapping_is_stable_under_query_and_donor_reordering():
    args = feature_inputs()
    first = build_eu_biology_features(*args, random_seed=17)
    changed = list(args)
    changed[2] = args[2][::-1]
    changed[3] = args[3][::-1]
    changed[4], changed[5], changed[6] = args[4][::-1], args[5][::-1], args[6][::-1]
    second = build_eu_biology_features(*changed, random_seed=17)
    np.testing.assert_allclose(first['values'], second['values'][::-1], atol=1e-14, rtol=1e-14)
    for a, b in zip(first['audit']['relation_diagnostics'], second['audit']['relation_diagnostics']):
        np.testing.assert_array_equal(a['randomized_similarity'], b['randomized_similarity'][::-1, ::-1])


@pytest.mark.parametrize('seed', [None, 7])
def test_query_future_poison_does_not_enter_features_or_random_matching(seed):
    args = feature_inputs()
    first = build_eu_biology_features(*args, random_seed=seed)
    modified = deepcopy(args)
    modified[0]['Y'][:, 1:] = np.nan
    second = build_eu_biology_features(*modified, random_seed=seed)
    np.testing.assert_array_equal(first['values'], second['values'])
    if seed is not None:
        for a, b in zip(first['audit']['relation_diagnostics'], second['audit']['relation_diagnostics']):
            np.testing.assert_array_equal(a['randomized_similarity'], b['randomized_similarity'])


@pytest.mark.parametrize('key,value', [('site', 'OTHER'), ('platform', 'OTHER'),
                                      ('cell_line', 'U2OS'), ('actual_dose_uM', 1.),
                                      ('exposure_hours_protocol_nominal', 48.)])
def test_hard_context_mismatch_returns_exact_zero(key, value):
    args = list(feature_inputs())
    args[1]['units'][0][key] = value
    for seed in (None, 8):
        out = build_eu_biology_features(*args, random_seed=seed)
        assert not out['support'][0]
        np.testing.assert_array_equal(out['values'][0], np.zeros(24))


def test_same_chemical_group_and_explicit_permissions_remain_excluded():
    args = list(feature_inputs())
    args[0]['groups'][3] = args[0]['groups'][0]
    allowed = np.ones((3, 10), bool)
    allowed[:, 1] = False
    out = build_eu_biology_features(*args, random_seed=15, allowed=allowed)
    for diag in out['audit']['relation_diagnostics']:
        assert diag['original_similarity'][0, 0] == 0
        assert diag['randomized_similarity'][0, 0] == 0
        np.testing.assert_array_equal(diag['randomized_similarity'][:, 1], np.zeros(3))


def _csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _workbook(path, worksheets):
    namespace = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relationship = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    with ZipFile(path, 'w') as archive:
        sheets = ''.join(f'<sheet name="{name}" sheetId="{j}" r:id="rId{j}"/>'
                         for j, name in enumerate(worksheets, 1))
        archive.writestr('xl/workbook.xml', f'<workbook xmlns="{namespace}" xmlns:r="{relationship}"><sheets>{sheets}</sheets></workbook>')
        links = ''.join(f'<Relationship Id="rId{j}" Target="worksheets/sheet{j}.xml"/>' for j in range(1, len(worksheets)+1))
        archive.writestr('xl/_rels/workbook.xml.rels', f'<Relationships>{links}</Relationships>')
        for j, rows in enumerate(worksheets.values(), 1):
            rendered = []
            for i, row in enumerate(rows, 1):
                cells = []
                for column, value in enumerate(row):
                    address = chr(ord('A')+column)+str(i)
                    if value is None:
                        continue
                    if isinstance(value, bool):
                        cells.append(f'<c r="{address}" t="b"><v>{int(value)}</v></c>')
                    else:
                        cells.append(f'<c r="{address}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
                rendered.append(f'<row r="{i}">{"".join(cells)}</row>')
            archive.writestr(f'xl/worksheets/sheet{j}.xml', f'<worksheet xmlns="{namespace}"><sheetData>{"".join(rendered)}</sheetData></worksheet>')


@pytest.fixture
def metadata_sources(tmp_path):
    identity = [dict(object_id=f'e{i}', pdid=f'p{i}', smiles=f'S{i}', inchi=f'I{i}',
                     inchikey=f'K{i}') for i in range(3)]
    _csv(tmp_path/'identity.csv', identity)
    compounds = [['pdid', 'name', 'smiles', 'inchi', 'inchikey']]
    for r in identity:
        compounds.append([r['pdid'], r['object_id'], r['smiles'], r['inchi'], r['inchikey']])
    targets = [['pdid', 'name', 'target_name', 'gene_name', 'human', 'moa', 'activity_cell']]
    targets.append(['p0', 'compound0', 'targetA', 'GENEA', '1', 'inhibitor', '5.8'])
    targets.append(['p0', 'compound0', 'nonhuman target', 'OtherGene', '0', 'agonist', '-'])
    targets.append(['p1', 'compound1', 'targetB', 'GENEB', True, 'inhibitor', '6.2'])
    targets.append(['p1', 'compound1', 'targetB', 'GENEB', True, 'agonist;antagonist', None])
    targets.append(['p2', 'compound2', '-', None, None, '-', None])
    _workbook(tmp_path/'annotations.xlsx', dict(COMPOUNDS=compounds, TARGETS=targets))
    rows = []
    for i in range(3):
        for j, role in enumerate(('X', 'Z1', 'Z2', 'V')):
            rows.append(dict(object_id=f'e{i}', resource_kind='FIT_COMPOUND',
                measurement_role=role, connectivity=f'g{i}', site='FMP', cell='HepG2',
                dose_record='10', dose_unit='uM', exposure_h_protocol='24',
                library_plate='B1', plate_uid=f'P{j}', well_position=f'A{i+1:02}',
                well_id=f'P{j}:A{i+1:02}'))
    _csv(tmp_path/'wells.csv', rows)
    (tmp_path/'phase.json').write_text(json.dumps(dict(allowed_compound_ids=['e0', 'e1', 'e2'])))
    return dict(identity_csv=tmp_path/'identity.csv', annotation_workbook=tmp_path/'annotations.xlsx',
                well_metadata_csv=tmp_path/'wells.csv', phase_manifest=tmp_path/'phase.json')


def test_metadata_target_action_identity_and_unknowns_are_preserved(metadata_sources):
    out = load_eu_biology_metadata(['e0', 'e1', 'e2'], ['g0', 'g1', 'g2'], **metadata_sources)
    arrays = out['arrays']
    np.testing.assert_array_equal(arrays['target_mask'], [True, True, False])
    np.testing.assert_array_equal(arrays['moa_mask'], [True, True, False])
    # Same action, different target does not fabricate same mechanism.
    assert arrays['moa'][0] @ arrays['moa'][1] == 0
    assert out['metadata']['target_names'] == ['targetA', 'targetB']
    assert any('agonist;antagonist' in n for n in out['metadata']['moa_names'])
    assert out['report']['biological_activity_values_used'] is False
    assert out['report']['measurements_read'] is False
    assert len(out['relations']) == 5
    assert out['metadata']['units'][0]['roles']['X']['cell_count'] is None
    assert out['metadata']['units'][0]['exact_execution_verified'] is False


def test_human_filter_is_explicit_not_silent_deletion(metadata_sources):
    out = load_eu_biology_metadata(['e0', 'e1'], ['g0', 'g1'], human_only=False, **metadata_sources)
    assert 'nonhuman target' in out['metadata']['target_names']
    assert out['report']['human_relation_counts']['nonhuman'] == 1


def test_identity_and_scope_mismatch_stop_before_data_use(metadata_sources):
    with pytest.raises(ValueError, match='allowlist'):
        load_eu_biology_metadata(['protected'], ['g4'], **metadata_sources)
    rows = list(csv.DictReader(metadata_sources['identity_csv'].open()))
    rows[0]['inchikey'] = 'wrong'
    _csv(metadata_sources['identity_csv'], rows)
    with pytest.raises(ValueError, match='identity mismatch'):
        load_eu_biology_metadata(['e0'], ['g0'], **metadata_sources)
