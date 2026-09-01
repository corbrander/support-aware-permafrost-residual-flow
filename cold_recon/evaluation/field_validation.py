from __future__ import annotations


def describe_real_data_validation() -> str:
    return (
        "Real public permafrost data usually lack complete 3D ground truth. "
        "Use hold-out observation prediction, borehole preservation, ERT/NMR "
        "consistency, and physical plausibility; use synthetic data for full-field metrics."
    )

