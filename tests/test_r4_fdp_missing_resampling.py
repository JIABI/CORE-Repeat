"""Missing-aware FDP block sensitivity using synthetic, fixed predictions only."""

import numpy as np

from opal2.r4_evaluation import campaign_resampling


MODES = ('fixed_list', 'new_campaign_topk')
POLICIES = ('CORE', 'HISTGB_CAL')
UTILITY_KEYS = (
    'difference_low', 'difference_high',
    'CORE_random_difference_low', 'CORE_random_difference_high',
)


def _fixture(*, missing=True, same_policy=False, one_block=False):
    n = 16
    ids = np.array([f'q{i:02}' for i in range(n)])
    gamma = np.linspace(-.2, .4, n)
    if missing:
        gamma[[1, 9]] = np.nan
    eligible = np.ones(n, dtype=bool)
    scores = {'CORE': -np.arange(n, dtype=float),
              'HISTGB_CAL': -np.roll(np.arange(n, dtype=float), 8)}
    if same_policy:
        scores['HISTGB_CAL'] = scores['CORE'].copy()
    policies = {}
    for name, score in scores.items():
        selected = np.zeros(n, dtype=bool)
        selected[np.argsort(-score)[:2]] = True
        policies[name] = dict(score=score, selected=selected)
    labels = np.array(['one'] * n if one_block else [f'b{i // 4}' for i in range(n)])
    return ids, gamma, eligible, policies, labels


def _interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return dict(valid_replicates=len(values),
                lower=float(np.quantile(values, .025)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                upper=float(np.quantile(values, .975)) if len(values) else None)


def _reference_replicates(ids, gamma, eligible, policies, labels, *, seed, repeats):
    """Independent transcription of frozen resampling and legacy value arithmetic.

    Selection counts and tie order match the pre-existing evaluator. FDP is
    separately computed by counting observed NULLs and unknown selected labels.
    No production missing-bound or interval helper is used here.
    """
    blocks = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(seed)
    out = {mode: {key: [] for key in UTILITY_KEYS} for mode in MODES}
    for mode in MODES:
        for name in POLICIES:
            out[mode][name] = dict(lower=[], upper=[], old_point=[], missing=0, undefined=0)

    def value_bounds(y, weights):
        known = np.isfinite(y)
        observed = float(np.dot(weights[known], y[known]))
        unknown = weights[~known]
        lo = observed + float(np.where(unknown >= 0, unknown * -1.02, unknown * .98).sum())
        hi = observed + float(np.where(unknown >= 0, unknown * .98, unknown * -1.02).sum())
        return lo / len(y), hi / len(y)

    for _ in range(repeats):
        ind = np.concatenate([blocks[b] for b in rng.integers(0, len(blocks), len(blocks))])
        y, e = gamma[ind], eligible[ind]
        k = min(len(ind) // 8, int(e.sum()))
        for mode in MODES:
            masks = {}
            for name in POLICIES:
                if mode == 'fixed_list':
                    chosen = policies[name]['selected'][ind]
                else:
                    rows = np.flatnonzero(e)
                    order = np.lexsort((rows, ids[ind][rows], -policies[name]['score'][ind][rows]))
                    chosen = np.zeros(len(ind), dtype=bool)
                    chosen[rows[order[:k]]] = True
                masks[name] = chosen
                selected_n = int(chosen.sum())
                risk = out[mode][name]
                if not selected_n:
                    risk['undefined'] += 1
                    continue
                unknown = int(np.sum(chosen & ~np.isfinite(y)))
                observed_null = int(np.sum(chosen & np.isfinite(y) & (y <= 0)))
                risk['lower'].append(observed_null / selected_n)
                risk['upper'].append((observed_null + unknown) / selected_n)
                risk['missing'] += int(unknown > 0)
                if not unknown:
                    risk['old_point'].append(float(np.mean(y[chosen] <= 0)))
            lo, hi = value_bounds(y, masks['CORE'].astype(float) - masks['HISTGB_CAL'].astype(float))
            out[mode]['difference_low'].append(lo)
            out[mode]['difference_high'].append(hi)
            random_weights = e.astype(float) * (int(masks['CORE'].sum()) / e.sum() if e.any() else 0.)
            lo, hi = value_bounds(y, masks['CORE'].astype(float) - random_weights)
            out[mode]['CORE_random_difference_low'].append(lo)
            out[mode]['CORE_random_difference_high'].append(hi)
    return out


def test_missing_selected_replicates_are_retained_with_counted_bounds():
    args = _fixture()
    seed, repeats = 41, 61
    actual = campaign_resampling(*args, seed=seed, repeats=repeats)
    expected = _reference_replicates(*args, seed=seed, repeats=repeats)
    assert actual['repeats'] == repeats
    for mode in MODES:
        for name in POLICIES:
            got, ref = actual['intervals'][mode][name + '_fdp'], expected[mode][name]
            assert ref['missing'] > 0
            assert got['valid_replicates'] == len(ref['lower'])
            assert got['valid_replicates'] > len(ref['old_point'])
            assert got['valid_replicates'] + got['undefined_replicates'] == repeats
            assert got['replicates_with_missing_selected'] == ref['missing']
            assert got['undefined_replicates'] == ref['undefined']
            assert got['lower_bound_distribution'] == _interval(ref['lower'])
            assert got['upper_bound_distribution'] == _interval(ref['upper'])
            assert got['lower'] == _interval(ref['lower'])['lower']
            assert got['upper'] == _interval(ref['upper'])['upper']


def test_no_missing_collapses_exactly_to_legacy_scalar_fdp_interval():
    args = _fixture(missing=False)
    seed, repeats = 12, 47
    actual = campaign_resampling(*args, seed=seed, repeats=repeats)
    expected = _reference_replicates(*args, seed=seed, repeats=repeats)
    for mode in MODES:
        for name in POLICIES:
            got = actual['intervals'][mode][name + '_fdp']
            legacy = _interval(expected[mode][name]['old_point'])
            assert got['lower_bound_distribution'] == got['upper_bound_distribution'] == legacy
            assert got['lower'] == legacy['lower']
            assert got['upper'] == legacy['upper']
            assert got['valid_replicates'] == legacy['valid_replicates']
            assert got['replicates_with_missing_selected'] == 0


def test_all_selected_outcomes_missing_has_full_zero_one_bounds():
    args = list(_fixture(one_block=True))
    args[1][:] = np.nan
    actual = campaign_resampling(*args, seed=7, repeats=5)
    for mode in MODES:
        for name in POLICIES:
            got = actual['intervals'][mode][name + '_fdp']
            assert (got['lower'], got['upper']) == (0., 1.)
            assert got['valid_replicates'] == got['replicates_with_missing_selected'] == 5
            assert got['undefined_replicates'] == 0
            assert got['lower_bound_distribution'] == _interval(np.zeros(5))
            assert got['upper_bound_distribution'] == _interval(np.ones(5))


def test_empty_selection_is_undefined_not_zero():
    args = list(_fixture(one_block=True))
    args[2][:] = False
    for values in args[3].values():
        values['selected'][:] = False
    actual = campaign_resampling(*args, seed=7, repeats=5)
    for mode in MODES:
        for name in POLICIES:
            got = actual['intervals'][mode][name + '_fdp']
            assert got['valid_replicates'] == got['replicates_with_missing_selected'] == 0
            assert got['undefined_replicates'] == 5
            assert got['lower'] is None and got['upper'] is None
            assert got['lower_bound_distribution'] == got['upper_bound_distribution'] == _interval([])


def test_identical_policies_have_identical_missing_aware_fdp():
    actual = campaign_resampling(*_fixture(same_policy=True), seed=63, repeats=39)
    for mode in MODES:
        assert actual['intervals'][mode]['CORE_fdp'] == actual['intervals'][mode]['HISTGB_CAL_fdp']


def test_missing_fix_preserves_inputs_and_all_legacy_utility_intervals_exactly():
    args = _fixture()
    snapshots = {name: {key: value.copy() for key, value in values.items()}
                 for name, values in args[3].items()}
    for values in args[3].values():
        for value in values.values():
            value.setflags(write=False)
    seed, repeats = 109, 53
    expected = _reference_replicates(*args, seed=seed, repeats=repeats)
    actual = campaign_resampling(*args, seed=seed, repeats=repeats)
    for mode in MODES:
        for key in UTILITY_KEYS:
            assert actual['intervals'][mode][key] == _interval(expected[mode][key])
    for name, values in args[3].items():
        for key, value in values.items():
            np.testing.assert_array_equal(value, snapshots[name][key])
