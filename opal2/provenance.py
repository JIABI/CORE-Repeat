"""Bind evaluation to the actual checkpoint fitting population, not new labels."""
from __future__ import annotations

import numpy as np


def bind_fitting_provenance(model, train_ids, validation_ids, *, pretraining_ids=(),
                            fitting_source_groups=None, fitting_sources_known=False,
                            pretraining_sources_unknown=False):
    train, validation = tuple(map(str, train_ids)), tuple(map(str, validation_ids))
    pretraining = tuple(map(str, pretraining_ids))
    if any(len(set(values)) != len(values) for values in (train, validation, pretraining)):
        raise ValueError("Duplicate checkpoint fitting identifiers")
    if (set(train) | set(pretraining)) & set(validation):
        raise ValueError("Training/pretraining and validation identifiers overlap")
    model.training_ids = train
    model.validation_ids = validation
    model.pretraining_ids = pretraining
    model.fitting_ids = frozenset(train + validation + pretraining)
    if fitting_source_groups is None:
        fitting_source_groups, fitting_sources_known = (), False
    if (type(fitting_sources_known) is not bool or type(pretraining_sources_unknown) is not bool or
            any(type(i) is not int or i < 0 for i in fitting_source_groups)):
        raise ValueError("Checkpoint fitting source provenance is invalid")
    model.fitting_source_groups = frozenset(fitting_source_groups)
    model.pretraining_sources_unknown = pretraining_sources_unknown
    model.fitting_sources_known = (fitting_sources_known and bool(model.fitting_source_groups)
                                   and not pretraining_sources_unknown)
    return model


def assert_evaluation_provenance(model, scaler, dataset, splits):
    train = tuple(dataset.ids[np.asarray(splits["train"])].tolist())
    validation = tuple(dataset.ids[np.asarray(splits["validation"])].tolist())
    if tuple(scaler.train_ids) != train:
        raise ValueError("Model scaler was fitted on a different training split")
    if hasattr(model, "fitting_ids"):
        if tuple(model.training_ids) != train or tuple(model.validation_ids) != validation:
            raise ValueError("Checkpoint training/validation split differs from supplied split")
        heldout = set(dataset.ids[np.r_[splits["calibration"], splits["evaluation"]]].tolist())
        if heldout & model.fitting_ids:
            raise ValueError("Checkpoint training or validation data leaked into evaluation")


def assert_unseen_outcomes(model, ids):
    if hasattr(model, "fitting_ids") and set(map(str, ids)) & set(model.fitting_ids):
        raise ValueError("Evaluation overlaps checkpoint training/validation/pretraining identifiers")
