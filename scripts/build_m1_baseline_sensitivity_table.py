from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_anchor_sensitivity"
    / "m1_anchor_sensitivity_test_id_detail.csv"
)
MODEL = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_seed41"
    / "m1_test_id_seed41_detail.csv"
)
GAUSSIAN = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_gaussian_seed430"
    / "m1_geostatistical_test_id_detail.csv"
)
LOGIT_GAUSSIAN = (
    ROOT
    / "outputs"
    / "m1_support_guided"
    / "formal_engineering_response_logit_gaussian_seed440"
    / "m1_geostatistical_logit_test_id_detail.csv"
)
PAPER_TABLE = (
    ROOT
    / "paper"
    / "engineering_geology_manuscript"
    / "M1_table7_baseline_sensitivity.md"
)
SOURCE = ROOT / "outputs" / "source_data" / "m1_baseline_sensitivity"


METHOD_ORDER = (
    "RF-24",
    "RF-100",
    "RF-300",
    "RF-500",
    "Extra Trees-300",
    "Gradient boosting",
    "Bounded Gaussian",
    "Logit-Gaussian",
    "Conditional residual flow (seed 41)",
)


def _bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, values.size, size=(5000, values.size))].mean(
        axis=1
    )
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(values)), float(lower), float(upper)


def _load() -> pd.DataFrame:
    missing = [str(path) for path in (ANCHOR, MODEL, GAUSSIAN, LOGIT_GAUSSIAN) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal baseline sensitivity artifacts: " + "; ".join(missing))

    anchor = pd.read_csv(ANCHOR)[["scene_id", "method", "eic_rmse"]].copy()

    model = pd.read_csv(MODEL)
    model_rows = pd.DataFrame(
        {
            "scene_id": model["scene_id"],
            "method": "Conditional residual flow (seed 41)",
            "eic_rmse": model["eic_rmse"],
        }
    )

    gaussian = pd.read_csv(GAUSSIAN)
    gaussian_rows = pd.DataFrame(
        {
            "scene_id": gaussian["scene_id"],
            "method": "Bounded Gaussian",
            "eic_rmse": gaussian["eic_rmse"],
        }
    )

    logit = pd.read_csv(LOGIT_GAUSSIAN)
    logit_rows = pd.DataFrame(
        {
            "scene_id": logit["scene_id"],
            "method": "Logit-Gaussian",
            "eic_rmse": logit["eic_rmse"],
        }
    )

    combined = pd.concat(
        [anchor, gaussian_rows, logit_rows, model_rows], ignore_index=True
    )
    combined = combined[combined["method"].isin(METHOD_ORDER)].copy()
    counts = combined.groupby("method")["scene_id"].nunique().to_dict()
    expected = {method: 100 for method in METHOD_ORDER}
    if counts != expected:
        raise RuntimeError(f"Expected 100 scenes for every method; found {counts}")
    return combined


def main() -> None:
    data = _load()
    pivot = data.pivot(index="scene_id", columns="method", values="eic_rmse")
    reference = pivot["RF-24"]
    rows: list[dict[str, float | int | str]] = []
    for index, method in enumerate(METHOD_ORDER):
        mean, lower, upper = _bootstrap_mean_ci(
            pivot[method].to_numpy(), 3100 + index
        )
        if method == "RF-24":
            difference_mean = difference_lower = difference_upper = float("nan")
        else:
            difference_mean, difference_lower, difference_upper = _bootstrap_mean_ci(
                (pivot[method] - reference).to_numpy(), 4100 + index
            )
        rows.append(
            {
                "method": method,
                "eic_rmse_mean": mean,
                "eic_rmse_ci95_lower": lower,
                "eic_rmse_ci95_upper": upper,
                "paired_difference_vs_rf24_mean": difference_mean,
                "paired_difference_vs_rf24_ci95_lower": difference_lower,
                "paired_difference_vs_rf24_ci95_upper": difference_upper,
                "n_scenes": int(pivot[method].notna().sum()),
            }
        )
    summary = pd.DataFrame(rows)

    lines = [
        "# Table 7. Strong-baseline and anchor-capacity sensitivity on the 100-scene ID set",
        "",
        "| Method | EIC RMSE, mean (95% scene-bootstrap CI) | Paired difference from RF-24 (95% CI) |",
        "|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rmse = (
            f"{row.eic_rmse_mean:.5f} "
            f"({row.eic_rmse_ci95_lower:.5f} to {row.eic_rmse_ci95_upper:.5f})"
        )
        if np.isfinite(row.paired_difference_vs_rf24_mean):
            difference = (
                f"{row.paired_difference_vs_rf24_mean:+.5f} "
                f"({row.paired_difference_vs_rf24_ci95_lower:+.5f} to "
                f"{row.paired_difference_vs_rf24_ci95_upper:+.5f})"
            )
        else:
            difference = "Reference"
        lines.append(f"| {row.method} | {rmse} | {difference} |")
    lines.extend(
        [
            "",
            "All rows use the same 100 immutable controlled scenes and scene-level resampling. "
            "They are a single-fit sensitivity analysis, not replacements for the primary "
            "three-training-seed comparison. Negative paired differences favour the listed method. "
            "The direct final-minus-Extra-Trees difference was -0.00034 (95% CI, -0.00165 to 0.00089). "
            "The transformed Gaussian is naturally bounded in EIC = 0--0.90; the primary bounded "
            "Gaussian contracts anomalies after conditioning.",
            "",
        ]
    )
    PAPER_TABLE.write_text("\n".join(lines), encoding="utf-8")
    SOURCE.mkdir(parents=True, exist_ok=True)
    data.to_csv(SOURCE / "baseline_sensitivity_detail.csv", index=False)
    summary.to_csv(SOURCE / "baseline_sensitivity_summary.csv", index=False)
    (SOURCE / "baseline_sensitivity_metadata.json").write_text(
        json.dumps(
            {
                "independent_unit": "controlled scene",
                "n_scenes": 100,
                "reference": "RF-24",
                "bootstrap_draws": 5000,
                "primary_three_seed_results_replaced": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(PAPER_TABLE)


if __name__ == "__main__":
    main()
