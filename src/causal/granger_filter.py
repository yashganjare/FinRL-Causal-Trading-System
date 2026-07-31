"""
granger_filter.py

Research-grade rolling Granger causality analysis.

Pipeline

master_features.parquet
        ↓
Stationarity Check
        ↓
Rolling Granger Analysis
        ↓
Feature Ranking
        ↓
causal_features.parquet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "master_features.parquet"

OUTPUT_FEATURE_FILE = DATA_DIR / "causal_features.parquet"

SCORE_FILE = RESULT_DIR / "granger_scores.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

# ==========================================================
# Load Dataset
# ==========================================================

def load_feature_dataset() -> pd.DataFrame:

    logger.info("Loading engineered feature dataset...")

    df = pd.read_parquet(INPUT_FILE)

    logger.info(f"Rows    : {len(df):,}")
    logger.info(f"Columns : {len(df.columns)}")

    return df


# ==========================================================
# Stationarity Test
# ==========================================================

def adf_test(
    series: pd.Series,
    alpha: float = 0.05,
):

    series = series.dropna()

    result = adfuller(series)

    statistic = result[0]
    p_value = result[1]

    stationary = p_value < alpha

    return {
        "statistic": statistic,
        "p_value": p_value,
        "stationary": stationary,
    }


# ==========================================================
# Automatic Stationarity Conversion
# ==========================================================

def ensure_stationary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Checking stationarity...")

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numeric_columns:

        result = adf_test(df[column])

        if result["stationary"]:

            logger.info(
                f"{column:<30} Stationary"
            )

        else:

            logger.info(
                f"{column:<30} Differencing"
            )

            df[column] = df[column].diff()

    df = df.dropna()

    return df


# ==========================================================
# Single Granger Test
# ==========================================================

def run_granger_test(
    data: pd.DataFrame,
    cause: str,
    target: str,
    max_lag: int = 5,
):

    try:

        subset = data[
            [target, cause]
        ].dropna()

        results = grangercausalitytests(
            subset,
            maxlag=max_lag,
            verbose=False,
        )

        lag_scores = []

        for lag in range(1, max_lag + 1):

            test = results[lag][0]["ssr_ftest"]

            lag_scores.append({

                "lag": lag,

                "f_stat": test[0],

                "p_value": test[1],

            })

        return lag_scores

    except Exception:

        return None


# ==========================================================
# Average Granger Score
# ==========================================================

def average_granger_score(
    lag_results: List[Dict],
):

    if lag_results is None:

        return None

    p_values = [
        x["p_value"]
        for x in lag_results
    ]

    f_values = [
        x["f_stat"]
        for x in lag_results
    ]

    return {

        "avg_p_value": float(
            np.mean(p_values)
        ),

        "best_p_value": float(
            np.min(p_values)
        ),

        "avg_f_stat": float(
            np.mean(f_values)
        ),

        "best_f_stat": float(
            np.max(f_values)
        ),
    }


# ==========================================================
# Candidate Feature Selection
# ==========================================================

def get_candidate_features(
    df: pd.DataFrame,
    target_column: str = "BTC_Return",
):

    candidates = []

    for col in df.columns:

        if col != target_column:

            if np.issubdtype(
                df[col].dtype,
                np.number,
            ):

                candidates.append(col)

    logger.info(
        f"Candidate Features : {len(candidates)}"
    )

    return candidates

    # ==========================================================
# Rolling Granger Analysis
# ==========================================================

def rolling_granger(
    df: pd.DataFrame,
    target_column: str,
    feature_column: str,
    window_size: int = 500,
    step_size: int = 100,
    max_lag: int = 5,
):

    logger.info(
        f"Rolling Granger: {feature_column} -> {target_column}"
    )

    window_scores = []

    for start in range(
        0,
        len(df) - window_size,
        step_size,
    ):

        end = start + window_size

        window = df.iloc[start:end]

        result = run_granger_test(
            window,
            cause=feature_column,
            target=target_column,
            max_lag=max_lag,
        )

        score = average_granger_score(result)

        if score is not None:

            window_scores.append(score)

    return window_scores


# ==========================================================
# Stability Score
# ==========================================================

def compute_stability_score(
    rolling_scores,
    alpha: float = 0.05,
):

    if len(rolling_scores) == 0:

        return 0.0

    significant = 0

    for score in rolling_scores:

        if score["best_p_value"] < alpha:

            significant += 1

    return significant / len(rolling_scores)


# ==========================================================
# Aggregate Feature Score
# ==========================================================

def compute_feature_score(
    feature_name: str,
    rolling_scores,
):

    if len(rolling_scores) == 0:

        return None

    avg_p = np.mean(
        [
            s["avg_p_value"]
            for s in rolling_scores
        ]
    )

    avg_f = np.mean(
        [
            s["avg_f_stat"]
            for s in rolling_scores
        ]
    )

    best_p = np.min(
        [
            s["best_p_value"]
            for s in rolling_scores
        ]
    )

    best_f = np.max(
        [
            s["best_f_stat"]
            for s in rolling_scores
        ]
    )

    stability = compute_stability_score(
        rolling_scores
    )

    return {

        "Feature": feature_name,

        "Average_P": avg_p,

        "Best_P": best_p,

        "Average_F": avg_f,

        "Best_F": best_f,

        "Stability": stability,

    }


# ==========================================================
# Evaluate All Features
# ==========================================================

def evaluate_all_features(
    df: pd.DataFrame,
    target_column: str = "BTC_Return",
):

    logger.info("=" * 60)
    logger.info("Running Rolling Granger Analysis")
    logger.info("=" * 60)

    candidates = get_candidate_features(
        df,
        target_column,
    )

    scores = []

    for feature in candidates:

        logger.info(f"Evaluating {feature}")

        rolling_scores = rolling_granger(
            df=df,
            target_column=target_column,
            feature_column=feature,
        )

        score = compute_feature_score(
            feature,
            rolling_scores,
        )

        if score is not None:

            scores.append(score)

    score_df = pd.DataFrame(scores)

    score_df = score_df.sort_values(
        by=[
            "Stability",
            "Average_F",
        ],
        ascending=False,
    )

    logger.info(
        f"Successfully evaluated {len(score_df)} features."
    )

    return score_df


# ==========================================================
# Select Top Features
# ==========================================================

def select_top_features(
    feature_scores: pd.DataFrame,
    top_k: int = 10,
):

    logger.info(
        f"Selecting Top {top_k} causal features..."
    )

    top = feature_scores.head(top_k)

    selected = top["Feature"].tolist()

    logger.info(
        f"Selected Features:\n{selected}"
    )

    return selected


# ==========================================================
# Build Causal Dataset
# ==========================================================

def build_causal_dataset(
    df: pd.DataFrame,
    selected_features,
    target_column: str = "BTC_Return",
):

    logger.info(
        "Building causal feature dataset..."
    )

    columns = selected_features + [
        target_column
    ]

    return df[columns].copy()

    # ==========================================================
# Save Results
# ==========================================================

def save_results(
    causal_df: pd.DataFrame,
    score_df: pd.DataFrame,
):

    logger.info("Saving causal dataset...")

    causal_df.to_parquet(
        OUTPUT_FEATURE_FILE,
        index=False,
    )

    logger.info(
        f"Saved -> {OUTPUT_FEATURE_FILE}"
    )

    score_df.to_csv(
        SCORE_FILE,
        index=False,
    )

    logger.info(
        f"Saved -> {SCORE_FILE}"
    )


# ==========================================================
# Display Summary
# ==========================================================

def print_summary(
    score_df: pd.DataFrame,
):

    logger.info("=" * 60)
    logger.info("Top Causal Features")
    logger.info("=" * 60)

    print(
        score_df.head(10).to_string(index=False)
    )

    logger.info("=" * 60)


# ==========================================================
# Main Pipeline
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("Starting Granger Causality Pipeline")
    logger.info("=" * 60)

    # ---------------------------------------
    # Load Dataset
    # ---------------------------------------

    df = load_feature_dataset()

    # ---------------------------------------
    # Stationarity
    # ---------------------------------------

    df = ensure_stationary(df)

    # ---------------------------------------
    # Evaluate Features
    # ---------------------------------------

    score_df = evaluate_all_features(
        df,
        target_column="BTC_Return",
    )

    # ---------------------------------------
    # Select Best Features
    # ---------------------------------------

    selected_features = select_top_features(
        score_df,
        top_k=10,
    )

    # ---------------------------------------
    # Create Final Dataset
    # ---------------------------------------

    causal_df = build_causal_dataset(
        df,
        selected_features,
        target_column="BTC_Return",
    )

    # ---------------------------------------
    # Save Outputs
    # ---------------------------------------

    save_results(
        causal_df,
        score_df,
    )

    # ---------------------------------------
    # Print Summary
    # ---------------------------------------

    print_summary(score_df)

    logger.info("=" * 60)
    logger.info("Pipeline Completed")
    logger.info("=" * 60)

    logger.info(
        f"Causal Dataset Shape : {causal_df.shape}"
    )

    logger.info(
        f"Selected Features : {len(selected_features)}"
    )


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":
    main()