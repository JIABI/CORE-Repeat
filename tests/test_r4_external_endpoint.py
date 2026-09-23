import json

import numpy as np
import pytest
from scipy.stats import spearmanr

from opal2.r4_external_endpoint import (
    bounded_contribution, common_anchor_arrays, external_endpoint,
    freeze_anchor_space, relation_vectors, spearman_rows, summarize_endpoint,
)


def _data(ids, values):
    values = np.asarray(values, float)
    return dict(ids=np.asarray(ids), groups=np.asarray(['group-'+i for i in ids]),
                Y=np.repeat(values[:, None, :], 4, axis=1),
                present=np.ones((len(ids), 4), bool), valid=np.ones((len(ids), 4), bool))


def test_average_tie_ranks_and_invalid_relations():
    x = np.array([[1, 2, 2, 4], [1, 1, 1, 1], [1, 2, np.nan, 3]])
    y = np.array([[4, 1, 2, 2], [1, 2, 3, 4], [1, 2, 3, 4]])
    actual = spearman_rows(x, y)
    assert actual[0] == pytest.approx(spearmanr(x[0], y[0]).statistic)
    assert np.isnan(actual[1:]).all()
    with pytest.raises(ValueError, match='three'):
        spearman_rows(x[:, :2], y[:, :2])


def test_common_anchors_use_technical_intersection_not_correlations():
    ids = ['b', 'a', 'c', 'd', 'e']
    fmp = _data(ids, [[1, 0], [0, 1], [1, 1], [-1, 2], [2, 1]])
    medina = _data(ids, [[-1, 0], [0, -1], [-1, -1], [1, -2], [-2, -1]])
    usc = _data(ids, [[1, 0], [0, 1], [1, 1], [-1, 2], [2, 1]])
    medina['present'][0, 0] = False
    usc['Y'][3, 2] = np.nan
    anchors, report = common_anchor_arrays(fmp, {'MEDINA': medina, 'USC': usc})
    assert list(anchors['ids']) == ['a', 'c', 'e']
    assert report['excluded_development_ids'] == ['b', 'd']
    np.testing.assert_allclose(anchors['MEDINA'], -anchors['FMP'])


def test_external_formula_uses_two_sites_and_all_four_validation_wells():
    anchors = dict(ids=np.array(['a', 'b', 'c', 'd']),
                   FMP=np.array([[1, 0], [0, 1], [-1, 0], [0, -1.]]))
    anchors.update(MEDINA=anchors['FMP']*3, USC=anchors['FMP']*7)
    x = np.array([[1, 0.], [1, 0.]])
    z = np.array([[[0, 2], [0, 2]], [[0, 2], [0, 2]]])
    external = {s: _data(['q', 'r'], [[0, 1], [0, 1]]) for s in ('MEDINA', 'USC')}
    external['USC']['valid'][1, 3] = False
    actual = external_endpoint(['q', 'r'], x, z, external, anchors)
    expected_before = spearmanr(relation_vectors(x[:1], anchors['FMP'])[0],
                               relation_vectors(np.array([[0, 1]]), anchors['USC'])[0]).statistic
    expected_after = spearmanr(relation_vectors((x+z.sum(1))[:1]/3, anchors['FMP'])[0],
                              relation_vectors(np.array([[0, 1]]), anchors['USC'])[0]).statistic
    assert actual['delta'][0] == pytest.approx(expected_after-expected_before)
    assert np.isnan(actual['delta'][1])
    assert np.isfinite(actual['MEDINA_delta'][1])
    assert np.isnan(actual['USC_delta'][1])


def test_paired_missing_actions_cancel_and_denominator_is_full_population():
    d = np.array([1, np.nan, np.nan, 0.])
    a, b = np.array([1, 1, 0, 0], bool), np.array([0, 1, 1, 0], bool)
    assert bounded_contribution(d, a.astype(float)-b.astype(float)) == dict(
        lower=-.25, upper=.75, point=None, unresolved_contributing_n=1, population_n=4)
    result = dict(delta=d, MEDINA_delta=d, USC_delta=d)
    summary = summarize_endpoint(result, {'CORE': {'selected': a}, 'HISTGB_CAL': {'selected': b}},
                                 np.ones(4, bool))
    assert summary['delta']['policies']['CORE']['lower'] == -.25
    assert summary['delta']['policies']['CORE']['upper'] == .75


def test_anchor_freeze_precedes_confirmation_x_and_uses_completed_dev_only(tmp_path):
    query = tmp_path/'query.npz'
    query.touch()
    with pytest.raises(PermissionError, match='before confirmation X'):
        freeze_anchor_space('not-read', {}, tmp_path/'blocked', confirmation_query=query)
    # Independent fixture with no confirmation file: exact DEV arrays only.
    ids = ['a', 'b', 'c', 'd']
    data = _data(ids, [[1, 0], [0, 1], [-1, 0], [0, -1.]])
    fmp = tmp_path/'dev.npz'
    np.savez(fmp, **data)
    directories = {}
    for site in ('MEDINA', 'USC'):
        directory = tmp_path/site
        directory.mkdir()
        (directory/'complete.json').write_text(json.dumps(dict(complete=True, stage='anchors', site=site)))
        np.savez(directory/'anchors.npz', **data)
        directories[site] = directory
    record = freeze_anchor_space(fmp, directories, tmp_path/'space', confirmation_query=tmp_path/'unopened.npz')
    assert record['available'] is True
    assert record['common_anchor_n'] == 4
    assert record['development_diagnostic_leave_self_out']['MEDINA']['mean'] == pytest.approx(1)
    assert record['confirmation_measurements_used'] is False
