from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cold_recon.evaluation.physics_consistency import (
    empirical_log_resistivity_np,
    empirical_unfrozen_water_np,
    facies_to_probability,
)
from cold_recon.evaluation.uncertainty import facies_entropy
from cold_recon.physics.settlement import settlement_potential_numpy


@dataclass(frozen=True)
class PhysicsRefinementConfig:
    temperature_min: float = -10.0
    temperature_max: float = 3.0
    heat_iterations: int = 16
    heat_strength: float = 0.35
    heat_anchor: float = 0.0
    unfrozen_weight: float = 0.7
    resistivity_weight: float = 0.4
    eic_min: float = 0.0
    eic_max: float = 0.75
    unfrozen_min: float = 0.0
    unfrozen_max: float = 0.8
    log_resistivity_min: float = 0.0
    log_resistivity_max: float = 12.0


def smooth_temperature_field(temperature: np.ndarray, cfg: PhysicsRefinementConfig) -> np.ndarray:
    temp = np.clip(np.asarray(temperature, dtype=np.float32), cfg.temperature_min, cfg.temperature_max)
    if cfg.heat_iterations <= 0 or min(temp.shape) < 3:
        return temp.astype(np.float32)
    base = temp.copy()
    out = temp.copy()
    strength = float(np.clip(cfg.heat_strength, 0.0, 1.0))
    anchor = float(np.clip(cfg.heat_anchor, 0.0, 1.0))
    for _ in range(int(cfg.heat_iterations)):
        prev = out
        proposal = prev.copy()
        neighbor_mean = (
            prev[:-2, 1:-1, 1:-1]
            + prev[2:, 1:-1, 1:-1]
            + prev[1:-1, :-2, 1:-1]
            + prev[1:-1, 2:, 1:-1]
            + prev[1:-1, 1:-1, :-2]
            + prev[1:-1, 1:-1, 2:]
        ) / 6.0
        proposal[1:-1, 1:-1, 1:-1] = (1.0 - strength) * prev[1:-1, 1:-1, 1:-1] + strength * neighbor_mean
        out = anchor * base + (1.0 - anchor) * proposal
    return np.clip(out, cfg.temperature_min, cfg.temperature_max).astype(np.float32)


def refine_realization(
    eic: np.ndarray,
    temperature: np.ndarray,
    unfrozen_water: np.ndarray,
    log_resistivity: np.ndarray,
    facies_probability: np.ndarray,
    cfg: PhysicsRefinementConfig | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or PhysicsRefinementConfig()
    eic_ref = np.clip(np.asarray(eic, dtype=np.float32), cfg.eic_min, cfg.eic_max)
    temp_ref = smooth_temperature_field(temperature, cfg)
    uw_original = np.clip(np.asarray(unfrozen_water, dtype=np.float32), cfg.unfrozen_min, cfg.unfrozen_max)
    uw_empirical = empirical_unfrozen_water_np(temp_ref, facies_probability)
    uw_ref = (1.0 - cfg.unfrozen_weight) * uw_original + cfg.unfrozen_weight * uw_empirical
    uw_ref = np.clip(uw_ref, cfg.unfrozen_min, cfg.unfrozen_max).astype(np.float32)
    rho_original = np.clip(np.asarray(log_resistivity, dtype=np.float32), cfg.log_resistivity_min, cfg.log_resistivity_max)
    rho_empirical = empirical_log_resistivity_np(eic_ref, temp_ref, uw_ref, facies_probability)
    rho_ref = (1.0 - cfg.resistivity_weight) * rho_original + cfg.resistivity_weight * rho_empirical
    rho_ref = np.clip(rho_ref, cfg.log_resistivity_min, cfg.log_resistivity_max).astype(np.float32)
    return {
        "eic": eic_ref.astype(np.float32),
        "temperature": temp_ref.astype(np.float32),
        "unfrozen_water": uw_ref,
        "log_resistivity": rho_ref,
        "resistivity": np.exp(np.clip(rho_ref, cfg.log_resistivity_min, cfg.log_resistivity_max)).astype(np.float32),
    }


def _field(data: dict[str, np.ndarray], sample_idx: int, key: str) -> np.ndarray:
    sample_key = f"{key}_samples"
    mean_key = f"{key}_mean"
    if sample_key in data:
        return np.asarray(data[sample_key][sample_idx])
    if mean_key in data:
        return np.asarray(data[mean_key])
    if key in data:
        return np.asarray(data[key])
    raise KeyError(f"Missing posterior field: {sample_key}, {mean_key}, or {key}")


def _summarize_samples(out: dict[str, np.ndarray], n_facies: int, dz: float | None) -> dict[str, np.ndarray]:
    for key in ["eic", "temperature", "unfrozen_water", "log_resistivity", "resistivity"]:
        samples = out[f"{key}_samples"]
        out[f"{key}_mean"] = samples.mean(axis=0).astype(np.float32)
        out[f"{key}_std"] = samples.std(axis=0).astype(np.float32)
    facies_samples = out["facies_samples"].astype(np.int64)
    probs = np.zeros((*facies_samples.shape[1:], n_facies), dtype=np.float32)
    for cls in range(n_facies):
        probs[..., cls] = np.mean(facies_samples == cls, axis=0)
    out["facies_probability"] = probs
    out["facies_entropy"] = facies_entropy(probs).astype(np.float32)
    out["facies_mode"] = np.argmax(probs, axis=-1).astype(np.int16)
    out["ice_rich_probability"] = np.mean(out["eic_samples"] > 0.30, axis=0).astype(np.float32)
    if dz is not None:
        out["settlement_potential"] = settlement_potential_numpy(
            out["eic_mean"],
            out["temperature_mean"] + 2.0,
            float(dz),
        )
    return out


def refine_posterior_dict(
    posterior: dict[str, np.ndarray],
    n_facies: int = 7,
    cfg: PhysicsRefinementConfig | None = None,
    dz: float | None = None,
) -> dict[str, np.ndarray]:
    cfg = cfg or PhysicsRefinementConfig()
    data = {key: np.asarray(value) for key, value in posterior.items()}
    out = dict(data)
    if "facies_samples" in data:
        facies_samples = data["facies_samples"].astype(np.int16)
        refined = {"eic": [], "temperature": [], "unfrozen_water": [], "log_resistivity": [], "resistivity": []}
        for sample_idx in range(facies_samples.shape[0]):
            facies_probs = facies_to_probability(facies_samples[sample_idx], n_facies=n_facies)
            fields = refine_realization(
                _field(data, sample_idx, "eic"),
                _field(data, sample_idx, "temperature"),
                _field(data, sample_idx, "unfrozen_water"),
                _field(data, sample_idx, "log_resistivity"),
                facies_probs,
                cfg=cfg,
            )
            for key, value in fields.items():
                refined[key].append(value)
        out["facies_samples"] = facies_samples
        for key, values in refined.items():
            out[f"{key}_samples"] = np.stack(values, axis=0).astype(np.float32)
        out = _summarize_samples(out, n_facies=n_facies, dz=dz)
    else:
        if "facies_probability" in data:
            facies_probs = data["facies_probability"].astype(np.float32)
        elif "facies_mode" in data:
            facies_probs = facies_to_probability(data["facies_mode"], n_facies=n_facies)
        elif "facies" in data:
            facies_probs = facies_to_probability(data["facies"], n_facies=n_facies)
        else:
            raise KeyError("Posterior must contain facies_samples, facies_probability, facies_mode, or facies.")
        fields = refine_realization(
            _field(data, 0, "eic"),
            _field(data, 0, "temperature"),
            _field(data, 0, "unfrozen_water"),
            _field(data, 0, "log_resistivity"),
            facies_probs,
            cfg=cfg,
        )
        for key, value in fields.items():
            out[f"{key}_mean"] = value
        out["facies_probability"] = facies_probs.astype(np.float32)
        out["facies_entropy"] = facies_entropy(facies_probs).astype(np.float32)
        out["facies_mode"] = np.argmax(facies_probs, axis=-1).astype(np.int16)
        out["ice_rich_probability"] = (out["eic_mean"] > 0.30).astype(np.float32)
        if dz is not None:
            out["settlement_potential"] = settlement_potential_numpy(out["eic_mean"], out["temperature_mean"] + 2.0, float(dz))
    out["physics_refinement_unfrozen_weight"] = np.asarray(cfg.unfrozen_weight, dtype=np.float32)
    out["physics_refinement_resistivity_weight"] = np.asarray(cfg.resistivity_weight, dtype=np.float32)
    out["physics_refinement_heat_iterations"] = np.asarray(cfg.heat_iterations, dtype=np.int32)
    out["physics_refinement_heat_strength"] = np.asarray(cfg.heat_strength, dtype=np.float32)
    out["physics_refinement_heat_anchor"] = np.asarray(cfg.heat_anchor, dtype=np.float32)
    return out
