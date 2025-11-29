from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.config import (
    TEST_FEATURES_PATH,
    TEST_1MIN_PATH,
    TEST_RAW_PATH,
    PROJECT_ROOT,
)

# =====================================================
# CONFIG
# =====================================================

EXP_NAME = "lightgbm_loho"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_all_homes.txt"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_lightgbm_loho.csv"


def select_feature_cols(x_test_full: pd.DataFrame) -> list[str]:
    """
    Select feature columns for prediction.

    For LOHO we didn't save an artifact with feature_cols, so we just
    drop meta columns and use the rest, same as in train_loho.py.
    """
    drop_cols = [c for c in ["datetime", "home_id"] if c in x_test_full.columns]
    feature_cols = [c for c in x_test_full.columns if c not in drop_cols]
    return feature_cols


def main():
    # -------------------------------------------------
    # 1) Load 1-minute test features + timestamps
    # -------------------------------------------------
    print(f"[LOHO PRED] Loading test features from: {TEST_FEATURES_PATH}")
    x_test_full = pd.read_parquet(TEST_FEATURES_PATH)

    print(f"[LOHO PRED] Loading resampled test (1min) from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)

    # Normalize datetime
    x_test_full["datetime"] = pd.to_datetime(x_test_full["datetime"], utc=True)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)

    # Select feature columns (same logic as train_loho)
    feature_cols = select_feature_cols(x_test_full)
    x_test = x_test_full[feature_cols]

    print("[LOHO PRED] Test (features) samples :", x_test.shape[0])
    print("[LOHO PRED] Num features            :", x_test.shape[1])

    # -------------------------------------------------
    # 2) Load LOHO all-homes model and predict on 1-min grid
    # -------------------------------------------------
    print(f"\n[LOHO PRED] Loading LOHO model from: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print("[LOHO PRED] Predicting on 1-minute grid...")
    y_pred_1min = booster.predict(x_test)
    y_pred_1min = np.clip(y_pred_1min, 0.0, None)

    if len(y_pred_1min) != len(test_1min):
        raise ValueError(
            f"Length mismatch between test_1min ({len(test_1min)}) "
            f"and predictions ({len(y_pred_1min)})."
        )

    test_1min = test_1min.copy()
    test_1min["fridge_pred"] = y_pred_1min

    print(
        "[LOHO PRED] Pred range on 1-min grid:",
        float(np.min(y_pred_1min)),
        "->",
        float(np.max(y_pred_1min)),
    )

    # -------------------------------------------------
    # 3) Align predictions back to raw test timestamps
    # -------------------------------------------------
    print(f"\n[LOHO PRED] Loading raw test from: {TEST_RAW_PATH}")
    raw_test = pd.read_csv(TEST_RAW_PATH)
    raw_test["datetime"] = pd.to_datetime(raw_test["datetime"], utc=True)

    raw_test = raw_test.sort_values("datetime")
    test_1min = test_1min.sort_values("datetime")

    print("[LOHO PRED] Min raw test ts :", raw_test["datetime"].min())
    print("[LOHO PRED] Max raw test ts :", raw_test["datetime"].max())
    print("[LOHO PRED] Min 1min test ts:", test_1min["datetime"].min())
    print("[LOHO PRED] Max 1min test ts:", test_1min["datetime"].max())

    aligned = pd.merge_asof(
        raw_test[["datetime"]],
        test_1min[["datetime", "fridge_pred"]],
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    # Backfill first (for times before first 1-min ts),
    # then forward-fill, then fill any remaining NaNs with 0.
    aligned["fridge_pred"] = aligned["fridge_pred"].bfill().ffill().fillna(0.0)

    # -------------------------------------------------
    # 4) Load sample_submission and write predictions
    # -------------------------------------------------
    print(f"\n[LOHO PRED] Loading sample submission from: {SAMPLE_SUB_PATH}")
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    if len(sub) != len(aligned):
        raise ValueError(
            f"Length mismatch: sample_submission has {len(sub)} rows, "
            f"raw_test/aligned has {len(aligned)} rows."
        )

    if "fridge" not in sub.columns:
        raise ValueError("sample_submission.csv must contain a 'fridge' column.")

    sub["fridge"] = aligned["fridge_pred"].values

    OUT_SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT_SUB_PATH, index=False)

    print("\n[LOHO PRED] SAVED SUBMISSION TO:")
    print(OUT_SUB_PATH)


if __name__ == "__main__":
    main()
