"""Replay the exact failed integration with the production forward verifier."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .gram_factor_verified import decode_draws
from .gram_geometry import gram_gains
from .gram_oof_experiment import decode_draws as old_decode
from .hierarchical_geometry import sample_joint_coordinates


def main():
    sys.dont_write_bytecode = True
    project = Path(__file__).resolve().parents[1]
    old = project/'runs/hierarchical_stability_20260914_v1'
    diagnostic = project/'runs/hierarchical_stability_failure_diagnostic_20260914_v1'
    output = project/'runs/hierarchical_stability_repair_retest_20260914_v1'
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    manifest = json.loads((old/'run_manifest.json').read_text())
    record = manifest['repetitions'][1]['folds'][4]
    folder = old/'repetitions/repeat_1/folds/fold_4'
    stats = json.loads((folder/'preprocessing.json').read_text())
    with np.load(folder/'arms/RIDGE_TRAINCV/test/u_predictions.npz', allow_pickle=False) as saved:
        ids, mean, covariance = saved['ids'], saved['mean_u'], saved['covariance_u']
    assert ids.tolist() == record['test_ids']
    np.testing.assert_array_equal(covariance, np.broadcast_to(covariance[0], covariance.shape))
    torch.set_num_threads(manifest['config']['threads'])
    raw = sample_joint_coordinates(mean, covariance[0], manifest['config']['samples'],
        record['seed']+200000)*np.asarray(stats['u_scale'])+np.asarray(stats['u_center'])
    gram, audit = decode_draws(raw, verify=True)
    assert audit['draw_object_count'] == 254000
    assert audit['failed_factor_draws_high_precision_verified'] == 540
    gains = gram_gains(torch.as_tensor(gram)).numpy()
    with np.load(diagnostic/'failed_draw_fixture.npz', allow_pickle=False) as fixture:
        index = fixture['indices']
        np.testing.assert_array_equal(raw[index[:, 0], index[:, 1]], fixture['native_u'])
        np.testing.assert_array_equal(gram[index[:, 0], index[:, 1]], fixture['direct_gram'])
        failed_gains = gains[index[:, 0], index[:, 1]]
        np.testing.assert_array_equal(failed_gains, fixture['float_gram_gains'])
        np.testing.assert_allclose(failed_gains, fixture['high_precision_gains'], rtol=0., atol=1e-9)
        null_flips = int(np.sum((failed_gains <= 0) != (fixture['high_precision_gains'] <= 0)))
        assert null_flips == 0
        good = fixture['good_native_u'][None]
        old_good, _ = old_decode(good)
        new_good, _ = decode_draws(good)
        np.testing.assert_array_equal(new_good, old_good)
        np.testing.assert_array_equal(new_good[0], fixture['good_original_gram'])
    result = dict(complete=True, n_objects=len(ids), samples=manifest['config']['samples'],
        source_predictions=str(folder/'arms/RIDGE_TRAINCV/test/u_predictions.npz'),
        old_failed_run=str(old), sampling_seed=record['seed']+200000,
        original_failed_coordinates_and_Gram_bitwise_unchanged=True,
        original_failed_gains_bitwise_unchanged=True, successful_control_Grams_bitwise_unchanged=True,
        NULL_sign_flips_vs_independent_high_precision=null_flips,
        no_actual_measurements_read=True, no_model_retraining=True,
        elapsed_seconds=time.monotonic()-started, audit=audit)
    with (output/'retest.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='audit'}, indent=2))
    print(json.dumps(dict(verified_draws=audit['draw_object_count'],
        high_precision_verified=audit['failed_factor_draws_high_precision_verified'],
        functional=audit['functional_consistency']), indent=2))


if __name__ == '__main__':
    main()
