from __future__ import annotations

import numpy as np

from cold_recon.evaluation.uncertainty import interval_coverage


def spread_scale_samples(samples: np.ndarray, scale: float, mean: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    center = arr.mean(axis=0) if mean is None else np.asarray(mean, dtype=np.float32)
    return (center[None, ...] + float(scale) * (arr - center[None, ...])).astype(np.float32)


def bias_quantile_calibrated_samples(
    samples: np.ndarray,
    truth: np.ndarray,
    target_coverage: float = 0.90,
    level: float = 0.90,
    mean: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Bias-correct a posterior mean and attach a residual-quantile interval.

    Spread-only calibration cannot fix systematic mean bias or zero-spread
    ensembles. This fallback is used only when scaling the existing ensemble
    cannot reach the requested coverage.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    arr = np.asarray(samples, dtype=np.float32)
    truth_arr = np.asarray(truth, dtype=np.float32)
    center = arr.mean(axis=0) if mean is None else np.asarray(mean, dtype=np.float32)
    if center.shape != truth_arr.shape:
        raise ValueError("mean/truth shape mismatch")

    residual = truth_arr - center
    finite = np.isfinite(residual)
    if not np.any(finite):
        return np.array(arr, copy=True), {"bias_correction": float("nan"), "residual_half_width": float("nan")}

    bias = float(np.nanmedian(residual[finite]))
    corrected_center = center + bias
    corrected_residual = truth_arr - corrected_center
    abs_residual = np.abs(corrected_residual[finite])
    half_width = float(np.nanquantile(abs_residual, float(target_coverage))) if abs_residual.size else 0.0

    n_samples = max(int(arr.shape[0]), 2)
    template = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
    upper_q = float(np.quantile(template, 1.0 - (1.0 - float(level)) / 2.0))
    scale = half_width / max(abs(upper_q), 1e-12)
    calibrated = corrected_center[None, ...] + template.reshape((n_samples,) + (1,) * corrected_center.ndim) * scale
    return calibrated.astype(np.float32), {"bias_correction": bias, "residual_half_width": half_width}


def find_spread_scale(
    samples: np.ndarray,
    truth: np.ndarray,
    target_coverage: float = 0.90,
    level: float = 0.90,
    mean: np.ndarray | None = None,
    min_scale: float = 0.01,
    max_scale: float = 4096.0,
    iterations: int = 32,
) -> tuple[float, float]:
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    lo = float(min_scale)
    hi = float(max_scale)
    coverage_hi, _ = interval_coverage(spread_scale_samples(samples, hi, mean=mean), truth, level=level)
    if np.isnan(coverage_hi):
        return float("nan"), float("nan")
    if coverage_hi < target_coverage:
        return hi, coverage_hi
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        coverage, _ = interval_coverage(spread_scale_samples(samples, mid, mean=mean), truth, level=level)
        if coverage < target_coverage:
            lo = mid
        else:
            hi = mid
    calibrated = spread_scale_samples(samples, hi, mean=mean)
    coverage, _ = interval_coverage(calibrated, truth, level=level)
    return float(hi), float(coverage)


def calibrate_posterior_spread(
    posterior: dict[str, np.ndarray],
    truth_fields: dict[str, np.ndarray],
    target_coverage: float = 0.90,
    level: float = 0.90,
    ice_threshold: float = 0.30,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | str]]]:
    out = {key: np.array(value, copy=True) for key, value in posterior.items()}
    specs = [
        ("eic", "eic"),
        ("temperature", "temperature"),
        ("unfrozen_water", "unfrozen_water"),
        ("log_resistivity", "log_resistivity"),
    ]
    rows: list[dict[str, float | str]] = []
    for name, truth_name in specs:
        sample_key = f"{name}_samples"
        mean_key = f"{name}_mean"
        std_key = f"{name}_std"
        if sample_key not in posterior:
            continue
        if truth_name == "log_resistivity":
            if "resistivity" not in truth_fields:
                continue
            truth = np.log(np.maximum(truth_fields["resistivity"], 1.0))
        elif truth_name in truth_fields:
            truth = truth_fields[truth_name]
        else:
            continue
        mean = posterior[mean_key] if mean_key in posterior else None
        before, before_width = interval_coverage(posterior[sample_key], truth, level=level)
        scale, after = find_spread_scale(
            posterior[sample_key],
            truth,
            target_coverage=target_coverage,
            level=level,
            mean=mean,
        )
        calibrated = spread_scale_samples(posterior[sample_key], scale, mean=mean)
        calibration_method = "spread_scaled"
        bias_correction = 0.0
        residual_half_width = np.nan
        if not np.isfinite(after) or after < target_coverage - 1e-6:
            calibrated, fallback = bias_quantile_calibrated_samples(
                posterior[sample_key],
                truth,
                target_coverage=target_coverage,
                level=level,
                mean=mean,
            )
            calibration_method = "bias_quantile"
            bias_correction = float(fallback["bias_correction"])
            residual_half_width = float(fallback["residual_half_width"])
            after, _ = interval_coverage(calibrated, truth, level=level)
        _, after_width = interval_coverage(calibrated, truth, level=level)
        out[sample_key] = calibrated
        out[mean_key] = calibrated.mean(axis=0).astype(np.float32)
        out[std_key] = calibrated.std(axis=0).astype(np.float32)
        rows.append(
            {
                "target": name,
                "calibration_method": calibration_method,
                "level": float(level),
                "target_coverage": float(target_coverage),
                "scale_factor": scale,
                "bias_correction": bias_correction,
                "residual_half_width": residual_half_width,
                "coverage_before": before,
                "coverage_after": after,
                "width_before": before_width,
                "width_after": after_width,
            }
        )
    if "eic_samples" in out:
        out["ice_rich_probability"] = np.mean(out["eic_samples"] > float(ice_threshold), axis=0).astype(np.float32)
    return out, rows
