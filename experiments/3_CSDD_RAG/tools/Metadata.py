from typing import Any, TYPE_CHECKING
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tools.Statistics import add_fdr, finite

if TYPE_CHECKING:
    from Data import TrainingData
    from Retrieval import RetrievalResult


class MetadataTool:
    def __init__(self, settings: dict[str, Any]):
        self.settings = settings

    def _regression(
        self,
        train: pd.DataFrame,
        outcome: pd.Series,
        test: pd.DataFrame,
    ) -> dict[str, Any]:
        columns = train.columns.tolist()
        encoder = ColumnTransformer([
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), columns),
        ])
        model = Pipeline([
            ("encoder", encoder),
            ("ridge", Ridge(alpha=float(self.settings["ridge_alpha"]))),
        ])
        folds = int(self.settings["regression_folds"])
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)
        predictions = cross_val_predict(model, train, outcome, cv=splitter)
        model.fit(train, outcome)
        test_prediction = float(model.predict(test)[0])
        residuals = outcome.to_numpy(dtype=float) - predictions
        residual_std = float(np.std(residuals, ddof=1))
        return {
            "model": "OneHotEncoder + Ridge",
            "alpha": float(self.settings["ridge_alpha"]),
            "cross_validation_folds": folds,
            "n_train": len(train),
            "input_fields": columns,
            "cross_validated_mae": float(mean_absolute_error(outcome, predictions)),
            "cross_validated_rmse": float(mean_squared_error(outcome, predictions) ** 0.5),
            "cross_validated_r2": float(r2_score(outcome, predictions)),
            "test_prediction": test_prediction,
            "residual_standard_deviation": residual_std,
            "approximate_95_percent_interval": [
                test_prediction - 1.96 * residual_std,
                test_prediction + 1.96 * residual_std,
            ],
            "fit_scope": "reference_models_only",
        }

    @staticmethod
    def _associations(
        train: pd.DataFrame,
        outcome: pd.Series,
        test: pd.DataFrame,
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for field in train.columns:
            test_value = test.iloc[0][field]
            same_mask = train[field].notna() & train[field].eq(test_value)
            same = outcome.loc[same_mask]
            other = outcome.loc[~same_mask]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, p_value = stats.ttest_ind(
                    same,
                    other,
                    equal_var=False,
                    nan_policy="omit",
                )
            rows.append({
                "field": field,
                "test_value": None if pd.isna(test_value) else str(test_value),
                "n_same": len(same),
                "n_other": len(other),
                "mean_effect_same": finite(same.mean()),
                "mean_effect_other": finite(other.mean()),
                "effect_difference_same_minus_other": finite(same.mean() - other.mean()),
                "welch_t": finite(statistic),
                "welch_p": finite(p_value),
                "effect_difference_95_ci_low": finite(
                    same.mean() - other.mean()
                    - 1.96 * np.sqrt(
                        same.var(ddof=1) / max(len(same), 1)
                        + other.var(ddof=1) / max(len(other), 1)
                    )
                ),
                "effect_difference_95_ci_high": finite(
                    same.mean() - other.mean()
                    + 1.96 * np.sqrt(
                        same.var(ddof=1) / max(len(same), 1)
                        + other.var(ddof=1) / max(len(other), 1)
                    )
                ),
            })
        table = pd.DataFrame(rows)
        table["welch_fdr"] = add_fdr(table, "welch_p")
        return table.sort_values("welch_fdr", na_position="last").reset_index(drop=True)

    def analyze(
        self,
        data: "TrainingData",
        retrieval: "RetrievalResult",
    ) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
        reference_ids = data.reference_effects.index.intersection(data.metadata.index)
        train = data.metadata.loc[reference_ids].copy()
        outcome = data.reference_effects.loc[reference_ids]
        test = data.metadata.reindex([data.test_model_id]).copy()
        associations = self._associations(train, outcome, test)
        regression = self._regression(train, outcome, test)
        regression["difference_from_knn_baseline"] = (
            float(regression["test_prediction"])
            - float(retrieval.baseline["weighted_prediction"])
        )
        package = {
            "scope": "Metadata group comparisons and regression use reference models only.",
            "test_missing_fraction": float(test.isna().mean(axis=1).iloc[0]),
            "test_group_associations": associations.to_dict(orient="records"),
            "ridge_regression": regression,
            "test_and_top_k_context": retrieval.metadata_context.to_dict(
                orient="records"
            ),
        }
        return package, {"metadata_associations": associations}
