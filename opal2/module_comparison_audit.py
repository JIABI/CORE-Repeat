"""Read-only paired optional-module audit around an unchanged cohort policy.

The caller supplies already frozen predictions and outcome-based evaluation
records. This module fits nothing, chooses no strength or gate, and writes no
files. Support-stratified scores are descriptive conditional performance, not
causal treatment effects. Whole-cohort top-k can displace unsupported objects.
"""
from __future__ import annotations

import numpy as np


def _mask(value, n, name):
    array = np.asarray(value)
    if array.dtype != bool or array.shape != (n,):
        raise ValueError(name + ' must be an aligned Boolean vector')
    return array


def audit_paired_module(*, ids, core_predictions, module_predictions,
                       eligible_support, active_mask, core_selected,
                       module_selected, actual_gamma, cell_ids=None):
    """Check off invariants and report paired scores and allocation spillover.

    Prediction mappings must contain only predictive quantities / proper scores,
    never selection-dependent policy values. Shared streams are required for
    exact equality on inactive rows. Cell IDs identify predeclared budget cells,
    not independent inferential units. No confidence bound is calculated here.
    """
    names = np.asarray(ids, str)
    if names.ndim != 1 or len(names) == 0 or np.any(names == '') or len(set(names)) != len(names):
        raise ValueError('Unique nonempty episode IDs are required')
    n = len(names)
    support = _mask(eligible_support, n, 'eligible_support')
    active = _mask(active_mask, n, 'active_mask')
    before = _mask(core_selected, n, 'core_selected')
    after = _mask(module_selected, n, 'module_selected')
    if np.any(active & ~support):
        raise ValueError('An unsupported object cannot activate the module')
    gamma = np.asarray(actual_gamma, float)
    if gamma.shape != (n,) or not np.isfinite(gamma).all():
        raise ValueError('Actual Gamma must be finite and aligned')
    labels = np.zeros(n, int) if cell_ids is None else np.asarray(cell_ids)
    if labels.shape != (n,):
        raise ValueError('Budget cell IDs must align with episode IDs')
    for cell in np.unique(labels):
        take = labels == cell
        if before[take].sum() != after[take].sum():
            raise ValueError('The two policies must have identical budgets in every declared cell')
    if not core_predictions or set(core_predictions) != set(module_predictions):
        raise ValueError('Matching nonempty predictive mappings are required')
    scores = {}
    any_prediction_changed = np.zeros(n, bool)
    for key in core_predictions:
        base = np.asarray(core_predictions[key])
        candidate = np.asarray(module_predictions[key])
        if (base.shape != candidate.shape or base.ndim < 1 or base.shape[0] != n
                or not np.isfinite(base).all() or not np.isfinite(candidate).all()):
            raise ValueError('Predictive arrays must be finite and aligned: ' + key)
        if not np.array_equal(base[~active], candidate[~active]):
            raise ValueError('Inactive predictive rows must exactly recover the core: ' + key)
        changed = (base != candidate).reshape(n, -1).any(1)
        any_prediction_changed |= changed
        difference = (candidate.astype(float) - base.astype(float)).reshape(n, -1).mean(1)
        scores[key] = dict(
            full_cohort_mean_difference=float(difference.mean()),
            supported_mean_difference=float(difference[support].mean()) if support.any() else None,
            active_mean_difference=float(difference[active].mean()) if active.any() else None,
            changed_objects=int(changed.sum()),
        )
    displacement = after.astype(int) - before.astype(int)
    changed_selection = displacement != 0
    selected_count = int(before.sum())
    selected_delta = float(np.dot(displacement, gamma))
    return dict(
        n=n, eligible_support_count=int(support.sum()), active_count=int(active.sum()),
        changed_prediction_objects=int(any_prediction_changed.sum()),
        exact_inactive_recovery=True, paired_predictive_scores=scores,
        selection=dict(
            selected_count=selected_count,
            symmetric_difference=int(changed_selection.sum()),
            supported_membership_changes=int((changed_selection & support).sum()),
            unsupported_membership_changes=int((changed_selection & ~support).sum()),
            active_membership_changes=int((changed_selection & active).sum()),
            inactive_membership_changes=int((changed_selection & ~active).sum()),
            total_actual_value_difference=selected_delta,
            selected_mean_actual_value_difference=selected_delta / selected_count if selected_count else None,
            full_cohort_actual_value_difference=selected_delta / n,
            actual_null_count_difference=int(((gamma <= 0) * displacement).sum()),
        ),
        support_estimand='Descriptive performance conditional on predecision reference support; not a causal treatment effect',
        spillover='Inactive predictions can remain exact while cohort top-k membership changes',
        scope='Fixed predictions and predeclared paired policies; no fitting, selection or uncertainty inference',
        formal_certificate=False,
    )
