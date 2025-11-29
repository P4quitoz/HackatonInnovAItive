import pandas as pd

from src.common.config import (
    TEST_1MIN_PATH,
    TEST_FEATURES_PATH,
)
from src.common.features import build_features


def main():
    print(f"Loading resampled test from: {TEST_1MIN_PATH}")
    df_test = pd.read_csv(TEST_1MIN_PATH)

    print("Building test features...")
    X_test, _ = build_features(df_test, is_train=False)

    print(f"Saving test features to: {TEST_FEATURES_PATH}")
    X_test.to_parquet(TEST_FEATURES_PATH)

    print("Done.")
    print(f"X_test shape: {X_test.shape}")


if __name__ == "__main__":
    main()
