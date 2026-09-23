"""Metadata-only engineering preflight for optional resource modules.

These helpers do not read measurements, fit models, inspect outcomes, select a
coefficient, or change an existing switch. ``ELIGIBLE_FOR_CALIBRATION`` means
that the declared resources meet necessary engineering prerequisites, not that
the module is active, useful, sufficiently powered, or statistically certified.
Admission remains the separately run ``fit_radial_switch`` selection procedure;
query-level support and exact core fallback remain the caller's responsibility.

Boolean inputs are explicit upstream attestations, not facts inferred here.
In particular, ``roles_isolated`` must cover IDs, chemistry groups and physical
measurement roles, with query outcomes excluded from all fitting/calibration.
Missing or malformed resource metadata returns NOT_APPLICABLE with reasons;
it does not cause a training attempt or a training exception.
"""
from __future__ import annotations


ELIGIBLE_FOR_CALIBRATION = "ELIGIBLE_FOR_CALIBRATION"
NOT_APPLICABLE = "NOT_APPLICABLE"
MIN_MODEL_FIT_GROUPS = 4
MIN_DIST_CAL_GROUPS = 3


def _flag(name, value, reasons, checks):
    # Do not turn strings such as "false", missing values, or counts into truth.
    verified = type(value) is bool
    checks[name] = value if verified else None
    if not verified:
        reasons.append(dict(code=f"{name}_unverified",
                            message=f"{name} requires explicit Boolean metadata."))
    elif not value:
        reasons.append(dict(code=f"{name}_not_satisfied",
                            message=f"Required resource condition {name} is false."))


def _groups(name, values, minimum, reasons, checks):
    valid = values is not None and not isinstance(values, (str, bytes, dict))
    try:
        entries = list(values) if valid else []
    except TypeError:
        valid, entries = False, []
    valid = valid and all(isinstance(v, str) and bool(v.strip()) for v in entries)
    if not valid:
        checks[name] = dict(records=None, unique_groups=None, required_minimum=minimum)
        reasons.append(dict(code=f"{name}_unverified",
                            message=f"{name} requires actual nonempty string group identifiers, not an asserted count."))
        return
    unique = len(set(entries))
    checks[name] = dict(records=len(entries), unique_groups=unique, required_minimum=minimum)
    if unique < minimum:
        reasons.append(dict(code=f"insufficient_{name}",
                            message=f"{name} has {unique} unique groups; the current procedure requires at least {minimum}."))
    return set(entries)


def _report(module, reasons, checks, prerequisites):
    return dict(
        module=module,
        status=NOT_APPLICABLE if reasons else ELIGIBLE_FOR_CALIBRATION,
        eligible_for_calibration=not reasons,
        active=False,
        reasons=reasons,
        checks=checks,
        prerequisites=prerequisites,
        next_step=("Resolve or document missing resources; retain the unchanged core."
                   if reasons else "Run the declared fit/calibration workflow; retain the core unless the separate calibrated switch admits the module."),
        limitations=[
            "Metadata-only necessary engineering conditions, not validation of underlying measurements or metadata truth.",
            "Minimum group counts are implementation requirements, not statistical-power or generalization claims.",
            "Eligibility is not activation; no coefficient, threshold, model, prediction, or current-run switch is changed.",
            "Per-query support, heldout evaluation and exact unsupported/disabled fallback are still required.",
        ],
    )


def representation_resource_eligibility(*, compatible_features=None,
        four_physical_repeat_roles=None, model_fit_groups=None,
        dist_cal_groups=None, roles_isolated=None, x_available=None):
    """Check resources for the current optional representation workflow.

    ``compatible_features`` attests ordered feature names and preprocessing
    semantics match the frozen core, not merely equal array widths.
    ``four_physical_repeat_roles`` attests real X/Z1/Z2/V well-level measurements
    in the declared fitting/calibration resources, not consensus, duplicated
    arrays, or four copies of one physical well. It does not authorize reading
    future query wells. ``x_available`` covers the required first-well inputs.

    Group sequences are the actual declared MODEL_FIT and DIST_CAL identifiers;
    repetitions are allowed and counted once per group. The four-group fit
    minimum matches ``fit_representation`` with its current default 0.2 group
    holdout (at least three INNER_FIT objects and one INNER_VALID object). This
    preflight does not validate custom optimizer/split configurations, profile
    finiteness, or successful optimization. A customized validation fraction
    requires a separate check of the actual internal split.

    The three-group DIST_CAL minimum matches ``fit_radial_switch``'s current
    three-fold grouped calibration. Roles must be declared isolated upstream.
    """
    reasons, checks = [], {}
    for name, value in (
        ("compatible_features", compatible_features),
        ("four_physical_repeat_roles", four_physical_repeat_roles),
        ("roles_isolated", roles_isolated),
        ("x_available", x_available),
    ):
        _flag(name, value, reasons, checks)
    fit_groups = _groups("model_fit_groups", model_fit_groups, MIN_MODEL_FIT_GROUPS, reasons, checks)
    cal_groups = _groups("dist_cal_groups", dist_cal_groups, MIN_DIST_CAL_GROUPS, reasons, checks)
    if fit_groups is not None and cal_groups is not None:
        shared = sorted(fit_groups & cal_groups)
        checks['model_fit_dist_cal_shared_groups'] = shared
        if shared:
            reasons.append(dict(code="model_fit_dist_cal_group_overlap",
                                message="Declared MODEL_FIT and DIST_CAL groups overlap despite requiring isolated roles."))
    return _report("representation", reasons, checks, dict(
        model_fit_minimum_unique_groups=MIN_MODEL_FIT_GROUPS,
        dist_cal_minimum_unique_groups=MIN_DIST_CAL_GROUPS,
        representation_source="conditional_state_representation.fit_representation; default validation_fraction=0.2",
        admission_source="optional_radial_modules.fit_radial_switch; grouped calibration, default splits=3",
    ))


def biology_resource_eligibility(*, legal_annotation_available=None,
        verified_context_match_to_reference_bank=None, roles_isolated=None,
        dist_cal_groups=None):
    """Check metadata resources for biological reference calibration.

    ``legal_annotation_available`` means that usable, provenance-supported
    target or MoA relationships can be constructed for the intended queries and
    actual reference bank, with identity, availability and vocabulary semantics
    respected. One valid channel suffices; known annotations with zero relation
    overlap are still legal metadata, not evidence to activate a module. It
    does not mean merely that a source table contains annotations.
    ``verified_context_match_to_reference_bank`` attests actual cell/time/dose
    compatibility for those intended query-reference relations; the relevant
    bank is the actual radial donor bank (DIST_CAL in current runners), not the
    frozen core's old 64-anchor bank. Unknown assay
    dose or a different time is not silently declared compatible. These are
    metadata booleans supplied by the caller, not inferred from structure.

    This is resource-level admission to calibration. It does not assert that
    every query has biological support, that relationships are predictive, or
    that a nonzero coefficient should be selected. It imposes no Tanimoto cutoff
    (including 0.7); target/MoA support need not be chemical-neighbor support.
    DIST_CAL needs the same three unique groups as the existing switch selector.
    """
    reasons, checks = [], {}
    for name, value in (
        ("legal_annotation_available", legal_annotation_available),
        ("verified_context_match_to_reference_bank", verified_context_match_to_reference_bank),
        ("roles_isolated", roles_isolated),
    ):
        _flag(name, value, reasons, checks)
    _groups("dist_cal_groups", dist_cal_groups, MIN_DIST_CAL_GROUPS, reasons, checks)
    return _report("biology", reasons, checks, dict(
        dist_cal_minimum_unique_groups=MIN_DIST_CAL_GROUPS,
        chemical_similarity_threshold=None,
        context_scope="actual intended query-reference bank relations, not dataset name alone",
        admission_source="optional_radial_modules.fit_radial_switch; grouped calibration, default splits=3",
    ))
