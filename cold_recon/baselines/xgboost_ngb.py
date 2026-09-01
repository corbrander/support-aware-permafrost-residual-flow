from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cold_recon.baselines.random_forest import _grid_points_and_surface, _surface_at_obs
from cold_recon.data.data_schema import OBS_TYPES, ObservationTable


@dataclass(frozen=True)
class GradientBoostingConfig:
    max_iter: int = 180
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    l2_regularization: float = 1e-3
    min_samples_leaf: int = 8
    random_state: int = 0


def _classifier(cfg: GradientBoostingConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=int(cfg.max_iter),
        learning_rate=float(cfg.learning_rate),
        max_leaf_nodes=int(cfg.max_leaf_nodes),
        l2_regularization=float(cfg.l2_regularization),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(cfg.random_state),
    )


def _regressor(cfg: GradientBoostingConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=int(cfg.max_iter),
        learning_rate=float(cfg.learning_rate),
        max_leaf_nodes=int(cfg.max_leaf_nodes),
        l2_regularization=float(cfg.l2_regularization),
        min_samples_leaf=int(cfg.min_samples_leaf),
        random_state=int(cfg.random_state),
    )


def reconstruct_gradient_boosting(
    sample: dict,
    n_facies: int = 7,
    config: GradientBoostingConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or GradientBoostingConfig()
    obs: ObservationTable = sample["observations"]
    query_features, _, shape = _grid_points_and_surface(sample)
    out: dict[str, np.ndarray] = {"gradient_boosting_backend": np.asarray("sklearn_hist_gradient_boosting")}

    facies_mask = obs.mask & (obs.type_ids == OBS_TYPES["borehole_facies"])
    if np.sum(facies_mask) >= 3:
        x_train = _surface_at_obs(sample, obs.coords[facies_mask])
        y_train = np.clip(obs.values[facies_mask].astype(np.int64), 0, n_facies - 1)
        clf = _classifier(cfg)
        clf.fit(x_train, y_train)
        facies = clf.predict(query_features).reshape(shape).astype(np.int16)
        out["facies"] = facies
        if hasattr(clf, "predict_proba"):
            proba_raw = clf.predict_proba(query_features)
            probs = np.zeros((query_features.shape[0], n_facies), dtype=np.float32)
            for col, cls in enumerate(clf.classes_.astype(np.int64)):
                if 0 <= cls < n_facies:
                    probs[:, cls] = proba_raw[:, col].astype(np.float32)
            denom = np.sum(probs, axis=1, keepdims=True)
            missing = denom[:, 0] <= 1e-6
            if np.any(missing):
                counts = np.bincount(y_train, minlength=n_facies).astype(np.float32)
                prior = counts / max(float(np.sum(counts)), 1.0)
                probs[missing] = prior[None, :]
                denom = np.sum(probs, axis=1, keepdims=True)
            out["facies_probability"] = (probs / np.maximum(denom, 1e-6)).reshape(*shape, n_facies).astype(np.float32)

    for type_name, field_name in [
        ("borehole_eic", "eic"),
        ("borehole_temperature", "temperature"),
        ("nmr_unfrozen_water", "unfrozen_water"),
        ("ert_log_resistivity", "log_resistivity"),
    ]:
        mask = obs.mask & (obs.type_ids == OBS_TYPES[type_name])
        if np.sum(mask) >= 6:
            x_train = _surface_at_obs(sample, obs.coords[mask])
            y_train = obs.values[mask].astype(np.float32)
            reg = _regressor(cfg)
            reg.fit(x_train, y_train)
            out[field_name] = reg.predict(query_features).reshape(shape).astype(np.float32)
    return out


class GradientBoostingBaseline:
    """XGBoost/NGBoost-style deterministic baseline using sklearn's histogram gradient boosting backend."""

    def __init__(self, config: GradientBoostingConfig | None = None, n_facies: int = 7) -> None:
        self.config = config or GradientBoostingConfig()
        self.n_facies = int(n_facies)
        self.prediction_: dict[str, np.ndarray] | None = None

    def fit(self, sample: dict) -> "GradientBoostingBaseline":
        self.prediction_ = reconstruct_gradient_boosting(sample, n_facies=self.n_facies, config=self.config)
        return self

    def predict(self) -> dict[str, np.ndarray]:
        if self.prediction_ is None:
            raise RuntimeError("GradientBoostingBaseline must be fit before predict().")
        return self.prediction_

    def reconstruct(self, sample: dict) -> dict[str, np.ndarray]:
        return reconstruct_gradient_boosting(sample, n_facies=self.n_facies, config=self.config)
