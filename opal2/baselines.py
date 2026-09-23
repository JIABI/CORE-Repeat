"""Real fitted comparators, with the same decision-time information boundary."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


@dataclass
class ConditionalResidualBootstrap:
    """Per-feature affine X-to-target prediction plus joint residual resampling.

    The residual vector is sampled as one complete multi-well measurement, so
    its within-well and between-well covariance is not destroyed. Training
    compounds only supply coefficients and residuals. This is a role-
    conditional baseline, not a fully batch-conditioned model.
    """
    ridge: float = 1.0

    def fit(self, x, targets):
        x, y = np.asarray(x, float), np.asarray(targets, float)
        if y.ndim != 3 or x.shape != (len(y), y.shape[2]) or len(x) < 2:
            raise ValueError("Expected X[N,D], targets[N,T,D]")
        self.x_mean = x.mean(0)
        self.y_mean = y.mean(0)
        dx = x - self.x_mean
        variance = np.mean(dx * dx, axis=0)
        self.slope = np.mean(dx[:, None, :] * (y - self.y_mean), axis=0) / (variance + self.ridge)
        fitted = self.y_mean + dx[:, None, :] * self.slope
        self.residuals = y - fitted
        return self

    def predict(self, x):
        return self.y_mean + (np.asarray(x) - self.x_mean)[:, None, :] * self.slope

    def sample_joint(self, x, n_samples, seed=0):
        rng = np.random.default_rng(seed)
        choices = rng.integers(len(self.residuals), size=(n_samples, len(x)))
        return self.predict(x)[None] + self.residuals[choices]


class DirectRepresentationHeads:
    """Complete 400-tree gain/NULL/POSITIVE heads on the SAME new representation.

    This is explicitly the representation-matched baseline, not a claim to
    reproduce OPAL 1.0's different feature preprocessing.
    """
    def __init__(self, seed=0, n_jobs=4):
        self.gain = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2,
                                       max_features="sqrt", random_state=seed, n_jobs=n_jobs)
        self.null = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                        max_features="sqrt", class_weight="balanced",
                                        random_state=seed+1, n_jobs=n_jobs)
        self.positive = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                            max_features="sqrt", class_weight="balanced",
                                            random_state=seed+2, n_jobs=n_jobs)

    def fit(self, representation, gamma):
        self.gain.fit(representation, gamma)
        self.null.fit(representation, np.asarray(gamma) <= 0)
        self.positive.fit(representation, np.asarray(gamma) >= .005)
        return self

    @staticmethod
    def probability(model, x):
        ix = np.flatnonzero(model.classes_ == True)
        return model.predict_proba(x)[:, ix[0]] if len(ix) else np.zeros(len(x))

    def predict(self, x):
        return {"mean": self.gain.predict(x), "p_null": self.probability(self.null, x),
                "p_positive": self.probability(self.positive, x)}
