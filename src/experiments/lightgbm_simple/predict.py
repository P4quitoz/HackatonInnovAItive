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

EXP_NAME = "lightgbm_simple"
EXPORT_DIR = PROJECT_ROOT / "exports" / EXP_NAME
MODEL_PATH = EXPORT_DIR / "model_final.txt"

SAMPLE_SUB_PATH = PROJECT_ROOT / "submission" / "sample_submission.csv"
OUT_SUB_PATH = PROJECT_ROOT / "submission" / "submission_lightgbm_simple.csv"


def main():
    # -------------------------------------------------
    # 1) Load 1-minute features + resampled timestamps
    # -------------------------------------------------
    print(f"Loading test features from: {TEST_FEATURES_PATH}")
    X_test_full = pd.read_parquet(TEST_FEATURES_PATH)

    print(f"Loading resampled test (1min) from: {TEST_1MIN_PATH}")
    test_1min = pd.read_csv(TEST_1MIN_PATH)

    # Normalise datetime
    X_test_full["datetime"] = pd.to_datetime(X_test_full["datetime"], utc=True)
    test_1min["datetime"] = pd.to_datetime(test_1min["datetime"], utc=True)

    # Same feature selection logic as in train.py
    drop_cols = [c for c in ["datetime", "home_id"] if c in X_test_full.columns]
    feature_cols = [c for c in X_test_full.columns if c not in drop_cols]
    X_test = X_test_full[feature_cols]

    print("Test (features) samples :", X_test.shape[0])
    print("Num features            :", X_test.shape[1])

    # -------------------------------------------------
    # 2) Load model and predict on 1-min grid
    # -------------------------------------------------
    print(f"\nLoading model from: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    print("Predicting on 1-minute grid...")
    y_pred_1min = booster.predict(X_test)
    y_pred_1min = np.clip(y_pred_1min, 0.0, None)

    if len(y_pred_1min) != len(test_1min):
        raise ValueError(
            f"Length mismatch between test_1min ({len(test_1min)}) "
            f"and predictions ({len(y_pred_1min)})."
        )

    test_1min = test_1min.copy()
    test_1min["fridge_pred"] = y_pred_1min

    print(
        "Pred range on 1-min grid:",
        float(np.min(y_pred_1min)),
        "→",
        float(np.max(y_pred_1min)),
    )

    # -------------------------------------------------
    # 3) Align back to raw test timestamps (only datetime)
    # -------------------------------------------------
    print(f"\nLoading raw test from: {TEST_RAW_PATH}")
    test_raw = pd.read_csv(TEST_RAW_PATH)
    test_raw["datetime"] = pd.to_datetime(test_raw["datetime"], utc=True)
    test_raw = test_raw.sort_values("datetime")

    print("Min raw test ts :", test_raw["datetime"].min())
    print("Min 1min test ts:", test_1min["datetime"].min())

    # merge_asof with backward direction: each raw ts gets the latest past 1-min ts
    aligned = pd.merge_asof(
        test_raw[["datetime"]],
        test_1min[["datetime", "fridge_pred"]],
        on="datetime",
        direction="backward",
        allow_exact_matches=True,
    )

    # IMPORTANT:
    # For early timestamps that fall *before* the first 1-min timestamp,
    # we now back-fill from the first prediction instead of forcing them to 0.
    aligned["fridge_pred"] = aligned["fridge_pred"].bfill().ffill().fillna(0.0)

    # -------------------------------------------------
    # 4) Load sample_submission and write predictions
    # -------------------------------------------------
    print(f"\nLoading sample submission from: {SAMPLE_SUB_PATH}")
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

    print("\nSAVED SUBMISSION TO:")
    print(OUT_SUB_PATH)


if __name__ == "__main__":
    main()
