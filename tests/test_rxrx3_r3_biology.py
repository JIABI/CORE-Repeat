import csv
import json
from copy import deepcopy

import numpy as np
import pytest

from opal2.rxrx3_r3_biology import load_rxrx3_biology_metadata, rxrx3_context_mask


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def sources(tmp_path):
    conditions, roles, identities = [], [], []
    for i in range(3):
        identities.append(dict(object_id=f'o{i}', original_id=f'drug{i}', smiles=f'S{i}',
                               perturbation_class='compound'))
        conditions.append(dict(condition_id=f'c{i}', object_id=f'o{i}', group_id=f'g{i}',
            identity_role='RXRX3_MODULE_DEV_CANDIDATE', measurement_access='True',
            roles_X_Z1_Z2_V_assigned='True', perturbation_type='COMPOUND',
            cell_record='HUVEC', dose_record='0.01', batch=f'batch{i}',
            treatment=f'drug{i}', smiles=f'S{i}', well_type_label=(
                'Control Compounds + Intron control' if i == 2 else 'Query Compounds + Intron control')))
        for j, role in enumerate(('X', 'Z1', 'Z2', 'V')):
            roles.append(dict(conditions[-1], role=role, physical_plate_id=f'batch{i}_P{j}',
                              well_id=f'batch{i}_P{j}_A01'))
    annotations = [dict(treatment='drug0', gene_symbol='G1', nM_value='1000',
                        measurement_type='ic50', database='ChEMBL'),
                   dict(treatment='drug0', gene_symbol='G2', nM_value='1001',
                        measurement_type='ec50', database='ChEMBL'),
                   dict(treatment='drug1', gene_symbol='G1', nM_value='3',
                        measurement_type='ec50, ic50', database='BindingDB'),
                   dict(treatment='drug2', gene_symbol='G3', nM_value='0',
                        measurement_type='ic50', database='ChEMBL')]
    paths = {}
    for name, rows in [('conditions', conditions), ('roles', roles),
                       ('identity', identities), ('annotation', annotations)]:
        paths[name+'_csv'] = tmp_path/(name+'.csv')
        write_rows(paths[name+'_csv'], rows)
    paths['r2_manifest'] = tmp_path/'manifest.json'
    paths['r2_manifest'].write_text(json.dumps(dict(ids=['c0', 'c1', 'c2'], groups=['g0', 'g1', 'g2'])))
    data = dict(ids=np.array(['c0', 'c1', 'c2']), groups=np.array(['g0', 'g1', 'g2']),
        object_ids=np.array(['o0', 'o1', 'o2']), dose=np.array([.01, .01, .01]),
        layout=np.array(['batch0', 'batch1', 'batch2']),
        well_ids=np.array([[f'batch{i}_P{j}_A01' for j in range(4)] for i in range(3)]),
        plates=np.array([[f'batch{i}_P{j}' for j in range(4)] for i in range(3)]))
    return data, paths


def test_fixed_prior_boundary_and_missing_moa_are_not_fabricated(sources):
    data, paths = sources
    out = load_rxrx3_biology_metadata(data, **paths)
    assert out['metadata']['target_names'] == ['G1']
    np.testing.assert_array_equal(out['arrays']['target'], [[1.], [1.], [0.]])
    np.testing.assert_array_equal(out['arrays']['target_mask'], [True, True, False])
    assert out['arrays']['moa'].shape == (3, 0)
    assert not out['arrays']['moa_mask'].any()
    assert out['report']['matched_annotation_rows'] == 4
    assert out['report']['eligible_annotation_rows'] == 2
    assert out['report']['no_HUVEC_potency_claim']
    assert out['report']['no_human_target_claim']
    assert all(r['human_status'] is None for r in out['relations'])


def test_loader_never_touches_outcomes_and_preserves_unknown_time_counts(sources):
    data, paths = sources

    class NoOutcomes(dict):
        def __getitem__(self, key):
            assert key != 'Y'
            return super().__getitem__(key)

    supplied = NoOutcomes(data, Y=object())
    out = load_rxrx3_biology_metadata(supplied, **paths)
    assert out['report']['measurements_read'] is False
    for unit in out['metadata']['units']:
        assert unit['time_exact_h'] is None
        assert unit['exposure_hours_protocol_nominal'] is None
        assert unit['exposure_hours_protocol_range'] == [18., 24.]
        assert all(role['cell_count'] is None for role in unit['roles'].values())


def test_context_allows_different_experiments_but_not_other_dose_protocol_background(sources):
    data, paths = sources
    metadata = load_rxrx3_biology_metadata(data, **paths)['metadata']
    assert rxrx3_context_mask(metadata, [0], [1, 2]).tolist() == [[True, True]]
    for field, value in [('actual_dose_uM', .1), ('protocol_id', 'other'),
                         ('protocol_max_h', 48.), ('background', 'CRISPR_gene'),
                         ('cell_line', 'U2OS'), ('platform', 'Other')]:
        changed = deepcopy(metadata)
        changed['units'][1][field] = value
        assert not rxrx3_context_mask(changed, [0], [1])[0, 0]
    metadata['units'][1]['protocol_min_h'] = None
    assert not rxrx3_context_mask(metadata, [0], [1])[0, 0]


@pytest.mark.parametrize('field,value,message', [
    ('ids', ['c0', 'c1', 'protected'], 'allowlist'),
    ('groups', ['g0', 'g1', 'wrong'], 'grouping'),
    ('object_ids', ['o0', 'o1', 'protected'], 'source compound'),
    ('dose', [.01, .1, .01], 'approved export'),
    ('layout', ['batch0', 'wrong', 'batch2'], 'approved export'),
])
def test_wrong_identity_or_condition_rejected(sources, field, value, message):
    data, paths = sources
    data[field] = np.asarray(value)
    with pytest.raises(ValueError, match=message):
        load_rxrx3_biology_metadata(data, **paths)


def test_wrong_physical_role_and_source_name_rejected(sources):
    data, paths = sources
    changed = deepcopy(data)
    changed['well_ids'][0] = changed['well_ids'][0, ::-1]
    with pytest.raises(ValueError, match='Physical role'):
        load_rxrx3_biology_metadata(changed, **paths)
    identities = list(csv.DictReader(paths['identity_csv'].open()))
    identities[0]['original_id'] = 'alias_not_exact'
    write_rows(paths['identity_csv'], identities)
    with pytest.raises(ValueError, match='Source identity'):
        load_rxrx3_biology_metadata(data, **paths)


def test_nonfinite_qualified_or_unrecognized_assay_values_do_not_become_targets(sources):
    data, paths = sources
    rows = list(csv.DictReader(paths['annotation_csv'].open()))
    rows.extend(dict(treatment='drug2', gene_symbol='bad'+str(i), nM_value=value,
                     measurement_type=kind, database='test')
                for i, (value, kind) in enumerate([('<100', 'ic50'), ('nan', 'ic50'),
                    ('inf', 'ic50'), ('-2', 'ic50'), ('2', 'unknown')]))
    write_rows(paths['annotation_csv'], rows)
    out = load_rxrx3_biology_metadata(data, **paths)
    assert out['metadata']['target_names'] == ['G1']
    assert not out['arrays']['target_mask'][2]


def test_allowlisted_subsets_and_metadata_row_reordering(sources):
    data, paths = sources
    first = load_rxrx3_biology_metadata(data, **paths)
    for field in ('conditions_csv', 'roles_csv', 'identity_csv'):
        rows = list(csv.DictReader(paths[field].open()))
        write_rows(paths[field], rows[::-1])
    subset = {key: value[[1, 0]] for key, value in data.items()}
    second = load_rxrx3_biology_metadata(subset, **paths)
    np.testing.assert_array_equal(second['arrays']['target'], first['arrays']['target'][[1, 0]])
    assert [u['id'] for u in second['metadata']['units']] == ['c1', 'c0']


def test_mixed_query_control_labels_keep_same_intron_background(sources):
    data, paths = sources
    rows = list(csv.DictReader(paths['roles_csv'].open()))
    rows[1]['well_type_label'] = 'Control Compounds + Intron control'
    write_rows(paths['roles_csv'], rows)
    out = load_rxrx3_biology_metadata(data, **paths)
    assert out['metadata']['units'][0]['roles']['Z1']['well_type_label'] == rows[1]['well_type_label']
    assert rxrx3_context_mask(out['metadata'], [0], [1])[0, 0]
    rows[1]['well_type_label'] = 'CRISPR guide'
    write_rows(paths['roles_csv'], rows)
    with pytest.raises(ValueError, match='Physical role'):
        load_rxrx3_biology_metadata(data, **paths)
