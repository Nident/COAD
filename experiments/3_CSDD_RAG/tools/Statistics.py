from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def finite(value: object) -> float | None:
    number = float(cast(Any, value))
    return number if np.isfinite(number) else None


def add_fdr(table: pd.DataFrame, p_column: str) -> pd.Series:
    adjusted = pd.Series(np.nan, index=table.index, dtype=float)
    mask = table[p_column].notna()
    adjusted.loc[mask] = multipletests(
        table.loc[mask, p_column].to_numpy(dtype=float),
        method="fdr_bh",
    )[1]
    return adjusted


def correlations(
    features: pd.DataFrame,
    outcome: pd.Series,
) -> pd.DataFrame:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    aligned_y = outcome.reindex(numeric.index).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = numeric.corrwith(aligned_y, method="pearson")
        spearman = numeric.corrwith(aligned_y, method="spearman")
    n = numeric.notna().mul(aligned_y.notna(), axis=0).sum()

    def p_values(coefficients: pd.Series) -> pd.Series:
        denominator = (1.0 - coefficients.pow(2)).clip(lower=1e-15)
        statistic = coefficients * np.sqrt((n - 2) / denominator)
        values = 2.0 * stats.t.sf(np.abs(statistic), (n - 2).clip(lower=1))
        return pd.Series(values, index=coefficients.index).where(n >= 3)

    table = pd.DataFrame({
        "feature": numeric.columns,
        "n": n.reindex(numeric.columns).to_numpy(dtype=int),
        "pearson_r": pearson.reindex(numeric.columns).to_numpy(dtype=float),
        "pearson_p": p_values(pearson).reindex(numeric.columns).to_numpy(dtype=float),
        "spearman_rho": spearman.reindex(numeric.columns).to_numpy(dtype=float),
        "spearman_p": p_values(spearman).reindex(numeric.columns).to_numpy(dtype=float),
    })
    table["pearson_fdr"] = add_fdr(table, "pearson_p")
    table["spearman_fdr"] = add_fdr(table, "spearman_p")
    table["best_fdr"] = table[["pearson_fdr", "spearman_fdr"]].min(axis=1)
    table["maximum_absolute_correlation"] = table[
        ["pearson_r", "spearman_rho"]
    ].abs().max(axis=1)
    clipped = table["pearson_r"].clip(-0.999999, 0.999999)
    fisher_z = np.arctanh(clipped)
    fisher_se = 1.0 / np.sqrt((table["n"] - 3).clip(lower=1))
    table["pearson_95_ci_low"] = np.tanh(fisher_z - 1.96 * fisher_se).where(
        table["n"] > 3
    )
    table["pearson_95_ci_high"] = np.tanh(fisher_z + 1.96 * fisher_se).where(
        table["n"] > 3
    )
    return table.sort_values(
        ["best_fdr", "maximum_absolute_correlation"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def ridge_summary(
    features: pd.DataFrame,
    outcome: pd.Series,
    test_features: pd.DataFrame,
    folds: int,
    alpha: float,
) -> dict[str, Any]:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    usable = numeric.columns[numeric.notna().any()].tolist()
    x = numeric[usable]
    y = outcome.reindex(x.index).astype(float)
    test = test_features.reindex(columns=usable).apply(pd.to_numeric, errors="coerce")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = cross_val_predict(model, x, y, cv=splitter)
    model.fit(x, y)
    test_prediction = float(model.predict(test)[0])
    residuals = y.to_numpy(dtype=float) - predictions
    residual_std = float(np.std(residuals, ddof=1))
    return {
        "model": "Ridge",
        "alpha": alpha,
        "cross_validation_folds": folds,
        "n_train": len(x),
        "n_features": len(usable),
        "cross_validated_mae": float(mean_absolute_error(y, predictions)),
        "cross_validated_rmse": float(mean_squared_error(y, predictions) ** 0.5),
        "cross_validated_r2": float(r2_score(y, predictions)),
        "test_prediction": test_prediction,
        "residual_standard_deviation": residual_std,
        "approximate_95_percent_interval": [
            test_prediction - 1.96 * residual_std,
            test_prediction + 1.96 * residual_std,
        ],
        "fit_scope": "reference_models_only",
    }


def binary_associations(
    features: pd.DataFrame,
    outcome: pd.Series,
) -> pd.DataFrame:
    y = outcome.reindex(features.index).astype(float)
    rows: list[dict[str, object]] = []
    for feature in features.columns:
        present = y.loc[features[feature].astype(bool)]
        absent = y.loc[~features[feature].astype(bool)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = stats.ttest_ind(
                present,
                absent,
                equal_var=False,
                nan_policy="omit",
            )
        pooled = np.sqrt(
            (
                (len(present) - 1) * present.var(ddof=1)
                + (len(absent) - 1) * absent.var(ddof=1)
            )
            / max(len(present) + len(absent) - 2, 1)
        )
        difference = float(present.mean() - absent.mean())
        standard_error = np.sqrt(
            present.var(ddof=1) / max(len(present), 1)
            + absent.var(ddof=1) / max(len(absent), 1)
        )
        rows.append({
            "feature": feature,
            "n_present": len(present),
            "n_absent": len(absent),
            "mean_effect_present": finite(present.mean()),
            "mean_effect_absent": finite(absent.mean()),
            "effect_difference_present_minus_absent": difference,
            "welch_t": finite(statistic),
            "welch_p": finite(p_value),
            "cohens_d": finite(difference / pooled) if pooled > 0 else None,
            "effect_difference_95_ci_low": finite(difference - 1.96 * standard_error),
            "effect_difference_95_ci_high": finite(difference + 1.96 * standard_error),
        })
    table = pd.DataFrame(rows)
    table["welch_fdr"] = add_fdr(table, "welch_p")
    return table.sort_values(
        ["welch_fdr", "effect_difference_present_minus_absent"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
