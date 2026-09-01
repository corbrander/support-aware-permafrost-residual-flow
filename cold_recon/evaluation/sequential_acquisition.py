from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class BoreholeCandidate:
    candidate_id: int
    x_index: int
    y_index: int


@dataclass(frozen=True)
class SequentialStep:
    policy: str
    step: int
    selected_candidate_id: int | None
    active_candidate_ids: tuple[int, ...]
    engineering_loss: float
    eic_rmse: float
    high_eic_error_rate: float
    mean_interval_width: float
    expected_decision_loss: float
    realized_decision_loss: float
    false_negative_rate: float
    false_positive_rate: float


def engineering_reconstruction_loss(
    posterior: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    high_eic_threshold: float = 0.30,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> dict[str, float]:
    eic_mean = np.asarray(posterior["eic_mean"], dtype=np.float64)
    eic_truth = np.asarray(truth["eic"], dtype=np.float64)
    rmse = float(np.sqrt(np.mean((eic_mean - eic_truth) ** 2)))
    truth_event = eic_truth >= float(high_eic_threshold)
    event_error = float(np.mean((eic_mean >= float(high_eic_threshold)) != truth_event))
    probability = np.asarray(
        posterior.get("ice_rich_probability", eic_mean >= float(high_eic_threshold)),
        dtype=np.float64,
    )
    probability = np.clip(probability, 0.0, 1.0)
    action_threshold = float(false_positive_cost) / (
        float(false_positive_cost) + float(false_negative_cost)
    )
    treatment_action = probability >= action_threshold
    false_negative = (~treatment_action) & truth_event
    false_positive = treatment_action & (~truth_event)
    realized_decision_loss = float(
        np.mean(
            float(false_negative_cost) * false_negative
            + float(false_positive_cost) * false_positive
        )
    )
    expected_decision_loss = float(
        np.mean(
            np.minimum(
                float(false_positive_cost) * (1.0 - probability),
                float(false_negative_cost) * probability,
            )
        )
    )
    if "temperature_mean" in posterior and "temperature" in truth:
        temperature_rmse = float(
            np.sqrt(
                np.mean(
                    (np.asarray(posterior["temperature_mean"]) - np.asarray(truth["temperature"])) ** 2
                )
            )
        )
    else:
        temperature_rmse = 0.0
    if "eic_std" in posterior:
        interval_width = float(2.0 * 1.645 * np.mean(np.asarray(posterior["eic_std"])))
    else:
        interval_width = float("nan")
    loss = (
        rmse
        + 0.50 * event_error
        + 0.05 * temperature_rmse
        + 0.20 * realized_decision_loss / max(float(false_negative_cost), 1.0e-6)
    )
    return {
        "engineering_loss": float(loss),
        "eic_rmse": rmse,
        "high_eic_error_rate": event_error,
        "mean_interval_width": interval_width,
        "expected_decision_loss": expected_decision_loss,
        "realized_decision_loss": realized_decision_loss,
        "false_negative_rate": float(np.mean(false_negative)),
        "false_positive_rate": float(np.mean(false_positive)),
    }


def _candidate_distance_scores(
    candidates: list[BoreholeCandidate],
    active_ids: set[int],
) -> np.ndarray:
    active = [candidate for candidate in candidates if candidate.candidate_id in active_ids]
    scores = np.zeros(len(candidates), dtype=np.float64)
    if not active:
        center_x = np.mean([candidate.x_index for candidate in candidates])
        center_y = np.mean([candidate.y_index for candidate in candidates])
        return -np.asarray(
            [np.hypot(candidate.x_index - center_x, candidate.y_index - center_y) for candidate in candidates]
        )
    for index, candidate in enumerate(candidates):
        scores[index] = min(
            np.hypot(candidate.x_index - other.x_index, candidate.y_index - other.y_index)
            for other in active
        )
    return scores


def acquisition_scores(
    policy: str,
    posterior: dict[str, np.ndarray],
    candidates: list[BoreholeCandidate],
    active_ids: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    scores = np.full(len(candidates), -np.inf, dtype=np.float64)
    available = np.asarray([candidate.candidate_id not in active_ids for candidate in candidates])
    if policy == "random":
        scores[available] = rng.random(np.sum(available))
        return scores
    distance = _candidate_distance_scores(candidates, active_ids)
    if policy in {"farthest", "grid_space_filling"}:
        scores[available] = distance[available]
        return scores

    def column_value(name: str, reducer: Callable[[np.ndarray], float]) -> np.ndarray:
        volume = np.asarray(posterior[name])
        return np.asarray(
            [reducer(volume[candidate.x_index, candidate.y_index]) for candidate in candidates],
            dtype=np.float64,
        )

    if policy == "variance":
        base = column_value("eic_std", np.nanmean)
    elif policy == "entropy":
        if "facies_entropy" in posterior:
            base = column_value("facies_entropy", np.nanmean)
        elif "lithology_entropy" in posterior:
            base = column_value("lithology_entropy", np.nanmean)
        else:
            probability = np.asarray(
                posterior.get("facies_probability", posterior["lithology_probability"]),
                dtype=np.float64,
            )
            entropy = -np.sum(probability * np.log(np.clip(probability, 1.0e-8, 1.0)), axis=-1)
            base = np.asarray(
                [np.nanmean(entropy[candidate.x_index, candidate.y_index]) for candidate in candidates]
            )
    elif policy == "high_eic_probability":
        base = column_value("ice_rich_probability", np.nanmax)
    elif policy in {"composite", "expected_loss"}:
        variance = column_value("eic_std", np.nanmean)
        event = column_value("ice_rich_probability", lambda x: np.nanmean(4.0 * x * (1.0 - x)))
        base = 0.55 * variance + 0.30 * event + 0.15 * distance / max(float(np.max(distance)), 1.0)
        if policy == "expected_loss" and "engineering_risk" in posterior:
            risk = column_value("engineering_risk", np.nanmean)
            base = base * (1.0 + np.maximum(risk, 0.0))
    else:
        raise ValueError(f"unknown acquisition policy: {policy}")
    scores[available] = base[available]
    return scores


def run_sequential_backtest(
    reconstruct: Callable[[tuple[int, ...]], dict[str, np.ndarray]],
    truth: dict[str, np.ndarray],
    candidates: Iterable[BoreholeCandidate],
    *,
    initial_candidate_ids: Iterable[int],
    additions: int = 5,
    policy: str = "composite",
    seed: int = 42,
) -> list[SequentialStep]:
    """Run actual reconstruct-select-update cycles for one scene and policy."""

    candidates = list(candidates)
    active = {int(value) for value in initial_candidate_ids}
    rng = np.random.default_rng(int(seed))
    rows: list[SequentialStep] = []
    for step in range(int(additions) + 1):
        posterior = reconstruct(tuple(sorted(active)))
        metrics = engineering_reconstruction_loss(posterior, truth)
        selected: int | None = None
        candidates_exhausted = False
        if step < int(additions):
            scores = acquisition_scores(policy, posterior, candidates, active, rng)
            if not np.any(np.isfinite(scores)):
                candidates_exhausted = True
            else:
                selected = int(candidates[int(np.nanargmax(scores))].candidate_id)
        rows.append(
            SequentialStep(
                policy=policy,
                step=step,
                selected_candidate_id=selected,
                active_candidate_ids=tuple(sorted(active)),
                **metrics,
            )
        )
        if candidates_exhausted:
            break
        if selected is not None:
            active.add(selected)
    return rows


def summarize_policy_regret(
    policy_rows: dict[str, list[SequentialStep]],
) -> list[dict[str, float | str]]:
    if not policy_rows:
        return []
    final_losses = {policy: rows[-1].engineering_loss for policy, rows in policy_rows.items() if rows}
    if not final_losses:
        return []
    oracle = min(final_losses.values())
    return [
        {
            "policy": policy,
            "initial_loss": rows[0].engineering_loss,
            "final_loss": rows[-1].engineering_loss,
            "loss_reduction": rows[0].engineering_loss - rows[-1].engineering_loss,
            "avoided_decision_loss": (
                rows[0].realized_decision_loss - rows[-1].realized_decision_loss
            ),
            "regret": rows[-1].engineering_loss - oracle,
        }
        for policy, rows in policy_rows.items()
        if rows
    ]
