from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests

from .access_policy import SplitManifest, build_split_manifest, load_split_manifest, require_scope
from .manifests import load_config, sha256_json, utc_now, write_json
from research.artifacts import ArtifactStore


class DataRepository:
    """Small, cached scientific repository. Protected labels are only returned by scoped methods."""

    META_FIELDS = ["ModelID", "PatientID", "CellLineName", "DepmapModelType", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype", "PrimaryOrMetastasis", "SampleCollectionSite", "Sex", "Age"]
    MUTATION_FIELDS = ["ModelID", "IsDefaultEntryForModel", "HugoSymbol", "EntrezGeneID", "ProteinChange", "VariantType", "MolecularConsequence", "VepImpact", "Hotspot", "LikelyLoF", "OncogeneHighImpact", "TumorSuppressorHighImpact", "HessDriver"]

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.config = load_config(self.config_path)
        data = self.config["data"]
        self.data_root = (self.config_path.parent / data["root"]).resolve()
        self.cache_root = (self.config_path.parent / data["cache_dir"]).resolve()
        self.cache_version = str(data.get("cache_version", "1"))
        self.artifact_root = (self.config_path.parent / self.config["artifacts"]["root"]).resolve()
        self._tables: dict[str, pd.DataFrame] = {}
        self._headers: dict[str, list[str]] = {}
        self._runs: dict[str, SplitManifest] = {}
        self._plans: dict[str, dict[str, Any]] = {}
        self._candidates: dict[str, dict[str, Any]] = {}
        self._predictions: dict[str, dict[str, Any]] = {}
        self._validation_submissions: dict[str, int] = {}

    @staticmethod
    def gene_symbol(value: str) -> str:
        return re.sub(r"\s+\(\d+\)$", "", str(value)).strip()

    @staticmethod
    def _is_missing_scalar(value: Any) -> bool:
        """Return a boolean only for scalar missing-value checks.

        ``pandas.isna`` is overloaded for both scalars and array-like values;
        the latter returns a Series/ndarray whose truth value is undefined.
        """
        result = pd.isna(value)
        return bool(result) if isinstance(result, (bool, np.bool_)) else False

    def _source(self, key: str) -> Path:
        filename = self.config["data"][key]
        return self.data_root / filename

    def _load(self, key: str, columns: list[str] | None = None) -> pd.DataFrame:
        if columns is None and key in self._tables:
            return self._tables[key]
        path = self.cache_root / f"{key}.parquet"
        if path.exists():
            frame = pd.read_parquet(path, columns=columns)
        else:
            frame = pd.read_csv(self._source(key), usecols=columns, low_memory=False) if columns else pd.read_csv(self._source(key), low_memory=False)
        if columns is None:
            self._tables[key] = frame
        return frame

    def _header(self, key: str) -> list[str]:
        if key not in self._headers:
            if (self.cache_root / f"{key}.parquet").exists():
                import pyarrow.parquet as pq
                self._headers[key] = pq.read_schema(self.cache_root / f"{key}.parquet").names
            else:
                self._headers[key] = pd.read_csv(self._source(key), nrows=0).columns.tolist()
        return self._headers[key]

    @staticmethod
    def _default_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if "IsDefaultEntryForModel" not in frame.columns:
            return frame
        mask = frame["IsDefaultEntryForModel"].astype(str).str.lower().isin({"yes", "true", "1"})
        return frame.loc[mask].copy()

    def _canonical_column(self, key: str, gene: str) -> str:
        symbol = self.gene_symbol(gene)
        candidates = [c for c in self._header(key) if self.gene_symbol(c) == symbol]
        if not candidates:
            raise KeyError(f"Gene {gene} is not present in {key}")
        return gene if gene in candidates else candidates[0]

    def _models(self) -> pd.DataFrame:
        frame = self._load("metadata").copy()
        return frame.drop_duplicates("ModelID").set_index("ModelID", drop=False)

    def initialize_run(self, target_gene: str, test_model_id: str, run_id: str) -> dict[str, Any]:
        model = self._models()
        effect = self._load("crispr", [self._header("crispr")[0]]).rename(columns={self._header("crispr")[0]: "ModelID"})
        model_ids = effect["ModelID"].astype(str).tolist()
        manifest_path = self.artifact_root / run_id / "split_manifest.json"
        if manifest_path.exists():
            manifest = load_split_manifest(manifest_path)
        else:
            manifest = build_split_manifest(run_id, model_ids, model, self.config, test_model_id, manifest_path)
        self._runs[run_id] = manifest
        return {"status": "OK", "run_id": run_id, "split_manifest_id": manifest.manifest_id, "split_manifest": manifest.as_dict(), "target_gene": self.gene_symbol(self._canonical_column("crispr", target_gene)), "test_model_id": test_model_id}

    def run(self, run_id: str) -> SplitManifest:
        if run_id not in self._runs:
            path = self.artifact_root / run_id / "split_manifest.json"
            if not path.exists():
                raise KeyError(f"Unknown run_id: {run_id}")
            self._runs[run_id] = load_split_manifest(path)
        return self._runs[run_id]

    def store(self, run_id: str) -> ArtifactStore:
        return ArtifactStore(self.artifact_root, run_id, self.cache_version)

    def manifest(self) -> dict[str, Any]:
        path = self.cache_root / "data_manifest.json"
        if path.exists():
            import json
            return json.loads(path.read_text(encoding="utf-8"))
        return {"status": "CACHE_NOT_PREPARED", "cache_version": self.cache_version, "sources": {k: str(self._source(k)) for k in self.config["data"] if k in {"crispr", "expression", "mutations", "cnv", "metadata"}}}

    def resolve_gene(self, gene: str) -> dict[str, Any]:
        result: dict[str, Any] = {"query": gene}
        for key in ("crispr", "expression", "cnv"):
            try:
                column = self._canonical_column(key, gene)
                result[key] = {"column": column, "symbol": self.gene_symbol(column), "available": True}
            except KeyError:
                result[key] = {"available": False}
        return result

    def _effects(self, target_gene: str, model_ids: Iterable[str]) -> pd.Series:
        column = self._canonical_column("crispr", target_gene)
        first = self._header("crispr")[0]
        frame = self._load("crispr", [first, column]).rename(columns={first: "ModelID"}).set_index("ModelID")
        return pd.to_numeric(frame[column], errors="coerce").reindex(list(model_ids))

    def target_summary(self, target_gene: str, run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        values = self._effects(target_gene, split.train).dropna()
        artifact = self.store(run_id).json("get_target_gene_summary", {"count": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=1)), "quantiles": {str(q): float(values.quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]}}, {"target_gene": target_gene}, "TRAIN")
        return {"status": "OK" if len(values) >= 3 else "INSUFFICIENT_DATA", "target_gene": self.gene_symbol(target_gene), "sample_size": len(values), "mean": float(values.mean()) if len(values) else None, "std": float(values.std(ddof=1)) if len(values) > 1 else None, "quantiles": {str(q): float(values.quantile(q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]} if len(values) else {}, "split_scope": "TRAIN", "artifact_id": artifact}

    def _expression(self, model_ids: list[str], genes: list[str]) -> pd.DataFrame:
        columns = ["ModelID", "IsDefaultEntryForModel", *[self._canonical_column("expression", gene) for gene in genes]]
        frame = self._default_rows(self._load("expression", list(dict.fromkeys(columns))))
        return frame.drop_duplicates("ModelID").set_index("ModelID").drop(columns=["IsDefaultEntryForModel", "IsDefaultEntryForMC"], errors="ignore").reindex(model_ids)

    def _cnv(self, model_ids: list[str], genes: list[str]) -> pd.DataFrame:
        columns = ["ModelID", "IsDefaultEntryForModel", *[self._canonical_column("cnv", gene) for gene in genes]]
        frame = self._default_rows(self._load("cnv", list(dict.fromkeys(columns))))
        return frame.drop_duplicates("ModelID").set_index("ModelID").drop(columns=["IsDefaultEntryForModel", "IsDefaultEntryForMC"], errors="ignore").reindex(model_ids)

    def _mutations(self, model_ids: list[str], genes: list[str], representation: str = "gene") -> pd.DataFrame:
        cols = [c for c in self.MUTATION_FIELDS if c in self._header("mutations")]
        frame = self._load("mutations", cols)
        frame = self._default_rows(frame)
        frame = frame[frame["ModelID"].astype(str).isin(set(model_ids))]
        symbols = {self.gene_symbol(g) for g in genes}
        frame = frame[frame["HugoSymbol"].astype(str).isin(symbols)]
        if representation == "variant":
            return frame.drop_duplicates().reset_index(drop=True)
        if frame.empty:
            return pd.DataFrame(0, index=model_ids, columns=sorted(symbols), dtype=float)
        result = pd.crosstab(frame["ModelID"].astype(str), frame["HugoSymbol"].astype(str)).clip(upper=1)
        return result.reindex(index=model_ids, columns=sorted(symbols), fill_value=0).astype(float)

    def test_profile(self, target_gene: str, test_model_id: str, modalities: list[str], run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        require_scope(split, [test_model_id], "TEST")
        profile: dict[str, Any] = {"ModelID": test_model_id, "target_gene": self.gene_symbol(target_gene), "split_scope": "TEST", "missingness": {}}
        for modality in modalities:
            # Accept common plural aliases produced by LLM tool calls while
            # keeping the repository's internal modality names canonical.
            modality = {"mutations": "mutation", "expressions": "expression", "copy_number": "cnv", "copy-number": "cnv"}.get(modality, modality)
            if modality == "expression":
                frame = self._expression([test_model_id], [target_gene])
                col = self._canonical_column("expression", target_gene)
                value = frame.iloc[0][col] if len(frame) else None
                profile[modality] = {self.gene_symbol(col): None if self._is_missing_scalar(value) else float(cast(Any, value))} if len(frame) else {}
            elif modality == "cnv":
                frame = self._cnv([test_model_id], [target_gene])
                col = self._canonical_column("cnv", target_gene)
                value = frame.iloc[0][col] if len(frame) else None
                profile[modality] = {self.gene_symbol(col): None if self._is_missing_scalar(value) else float(cast(Any, value))}
            elif modality == "mutation":
                profile[modality] = {self.gene_symbol(target_gene): bool(self._mutations([test_model_id], [target_gene]).iloc[0].iloc[0])}
            elif modality == "metadata":
                row = self._models().loc[test_model_id]
                profile[modality] = {key: (None if self._is_missing_scalar(row.get(key)) else str(row.get(key))) for key in self.META_FIELDS if key != "ModelID" and key in row.index}
            else:
                raise ValueError(f"Unsupported modality: {modality}")
        return {"status": "OK", "profile": profile, "protected_fields": ["GeneEffect", "target_gene_gene_effect"]}

    def page(self, frame: pd.DataFrame, page: int, page_size: int) -> dict[str, Any]:
        page_size = max(1, min(int(page_size), 1000))
        page = max(0, int(page))
        selected = frame.iloc[page * page_size : (page + 1) * page_size]
        records = cast(
            list[dict[str, Any]],
            selected.where(pd.notna(selected), None).to_dict(orient="records"),
        )
        return {"rows": records, "page": page, "page_size": page_size, "total_rows": len(frame)}

    def get_models(self, model_ids: list[str], fields: list[str], run_id: str, page: int, page_size: int) -> dict[str, Any]:
        self.run(run_id)
        frame = self._models().reindex(model_ids)
        fields = [f for f in fields if f in frame.columns] or ["ModelID", "OncotreeLineage"]
        return self.page(frame[fields].reset_index(drop=True), page, page_size)

    def get_expression(self, genes: list[str], model_ids: list[str], run_id: str, page: int, page_size: int) -> dict[str, Any]:
        self.run(run_id)
        frame = self._expression(model_ids, genes).reset_index()
        return self.page(frame, page, page_size)

    def get_cnv(self, genes: list[str], model_ids: list[str], run_id: str, page: int, page_size: int) -> dict[str, Any]:
        self.run(run_id)
        return self.page(self._cnv(model_ids, genes).reset_index(), page, page_size)

    def get_mutations(self, genes: list[str], model_ids: list[str], representation: str, run_id: str, page: int, page_size: int) -> dict[str, Any]:
        self.run(run_id)
        return self.page(self._mutations(model_ids, genes, representation).reset_index() if representation != "variant" else self._mutations(model_ids, genes, representation), page, page_size)

    def get_metadata(self, model_ids: list[str], fields: list[str], run_id: str, page: int, page_size: int) -> dict[str, Any]:
        self.run(run_id)
        frame = self._models().reindex(model_ids)
        fields = [f for f in fields if f in frame.columns] or ["ModelID", "OncotreeLineage"]
        return self.page(frame[fields].reset_index(drop=True), page, page_size)

    def train_effect(self, target_gene: str, model_ids: list[str] | None, run_id: str, page: int, page_size: int) -> dict[str, Any]:
        split = self.run(run_id)
        ids = model_ids or list(split.train)
        require_scope(split, ids, "TRAIN")
        frame = self._effects(target_gene, ids).rename("GeneEffect").reset_index().rename(columns={"index": "ModelID"})
        return self.page(frame, page, page_size)

    def _screen(self, target_gene: str, run_id: str, modality: str, genes: list[str] | None = None, top_k: int = 20) -> dict[str, Any]:
        split = self.run(run_id)
        y = self._effects(target_gene, split.train)
        if modality == "expression":
            header = [self._canonical_column("expression", g) for g in genes] if genes else [c for c in self._header("expression") if re.search(r"\(\d+\)$", str(c))]
            x = self._expression(list(split.train), header)
        elif modality == "cnv":
            header = [self._canonical_column("cnv", g) for g in genes] if genes else [c for c in self._header("cnv") if re.search(r"\(\d+\)$", str(c))]
            x = self._cnv(list(split.train), header)
        else:
            header = genes or []
            x = self._mutations(list(split.train), header)
        numeric = x.apply(pd.to_numeric, errors="coerce")
        aligned_y = pd.to_numeric(y.reindex(numeric.index), errors="coerce")
        # Spearman correlation is Pearson correlation on ranks.  Computing the
        # rank matrix once avoids one scipy call per ~20k gene columns.
        ranked = numeric.rank(axis=0)
        rho_series = ranked.corrwith(aligned_y.rank())
        n_series = numeric.notna().mul(aligned_y.notna(), axis=0).sum()
        rows: list[dict[str, Any]] = []
        for column, rho in rho_series.items():
            n_obs = int(n_series.get(column, 0))
            if n_obs < 8 or not np.isfinite(rho):
                continue
            denominator = max(1.0 - float(rho) ** 2, 1e-12)
            statistic = float(rho) * np.sqrt(max(n_obs - 2, 1) / denominator)
            p_value = float(2.0 * stats.t.sf(abs(statistic), max(n_obs - 2, 1)))
            rows.append({"feature": str(column), "gene": self.gene_symbol(str(column)), "method": "spearman", "estimate": float(rho), "p_value": p_value, "sample_size": n_obs, "effect_size": abs(float(rho))})
        table = pd.DataFrame(rows)
        if table.empty:
            return {"status": "INSUFFICIENT_DATA", "rows": [], "split_scope": "TRAIN"}
        table["adjusted_p_value"] = multipletests(table["p_value"].fillna(1.0), method="fdr_bh")[1]
        table = table.sort_values(["adjusted_p_value", "effect_size"], ascending=[True, False]).head(max(1, min(top_k, 100)))
        artifact = self.store(run_id).table(f"screen_{modality}_dependency_associations", table, {"target_gene": target_gene, "modality": modality, "top_k": top_k}, "TRAIN")
        table_rows = cast(list[dict[str, Any]], table.to_dict(orient="records"))
        return {"status": "OK", "modality": modality, "rows": table_rows, "split_scope": "TRAIN", "artifact_id": artifact, "tested_features": len(rows), "fdr_method": "benjamini-hochberg"}

    def screen_expression(self, target_gene: str, run_id: str, top_k: int = 20) -> dict[str, Any]:
        return self._screen(target_gene, run_id, "expression", top_k=top_k)

    def calculate_coexpression(self, gene: str, candidate_genes: list[str], run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        frame = self._expression(list(split.train), [gene, *candidate_genes])
        rows = []
        for other in frame.columns[1:]:
            pair = frame[[frame.columns[0], other]].dropna()
            if len(pair) >= 3:
                rho_raw, p_raw = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rho = float(cast(Any, rho_raw))
                p_value = float(cast(Any, p_raw))
            else:
                rho = float("nan")
                p_value = float("nan")
            rows.append({"gene": self.gene_symbol(other), "estimate": rho if np.isfinite(rho) else None, "p_value": p_value if np.isfinite(p_value) else None, "sample_size": len(pair), "relationship": "coexpression"})
        return {"status": "OK", "rows": rows, "split_scope": "TRAIN", "artifact_id": self.store(run_id).json("calculate_coexpression", rows, {"gene": gene, "candidate_genes": candidate_genes}, "TRAIN")}

    def describe_distribution(self, target_gene: str, run_id: str, scope: str = "TRAIN") -> dict[str, Any]:
        ids = self.run(run_id).scope(scope)
        values = self._effects(target_gene, ids).dropna()
        return {"status": "OK" if len(values) >= 3 else "INSUFFICIENT_DATA", "scope": scope, "sample_size": len(values), "mean": float(values.mean()) if len(values) else None, "median": float(values.median()) if len(values) else None, "std": float(values.std(ddof=1)) if len(values) > 1 else None, "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None}

    def metadata_associations(self, target_gene: str, run_id: str, field: str) -> dict[str, Any]:
        split = self.run(run_id)
        if field not in self._models().columns:
            return {"status": "INSUFFICIENT_DATA", "warnings": [f"Unknown metadata field: {field}"]}
        ids = list(split.train)
        y = self._effects(target_gene, ids)
        frame = self._models().loc[ids]
        rows = []
        for value, group in frame.groupby(field, dropna=False):
            outcome = y.reindex(group.index).dropna()
            rows.append({"field": field, "value": None if self._is_missing_scalar(value) else str(value), "sample_size": len(outcome), "mean_gene_effect": float(outcome.mean()) if len(outcome) else None})
        return {"status": "OK", "rows": rows, "split_scope": "TRAIN", "relationship": "metadata_stratification"}

    def _feature_frame(self, plan: dict[str, Any], model_ids: list[str]) -> pd.DataFrame:
        modality = str(plan["strategy"])
        features = plan.get("features_by_modality", {})
        if modality in {"expression_ridge", "cnv_ridge", "mutation_ridge", "multiomics_knn"}:
            pieces: list[pd.DataFrame] = []
            if modality in {"expression_ridge", "multiomics_knn"} and features.get("expression"):
                frame = self._expression(model_ids, list(features["expression"])).copy()
                frame.columns = [f"expression::{self.gene_symbol(c)}" for c in frame.columns]
                pieces.append(frame)
            if modality in {"cnv_ridge", "multiomics_knn"} and features.get("cnv"):
                frame = self._cnv(model_ids, list(features["cnv"])).copy()
                frame.columns = [f"cnv::{self.gene_symbol(c)}" for c in frame.columns]
                pieces.append(frame)
            if modality == "mutation_ridge" and features.get("mutation"):
                frame = self._mutations(model_ids, list(features["mutation"])).copy()
                frame.columns = [f"mutation::{c}" for c in frame.columns]
                pieces.append(frame)
            return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=model_ids)
        if modality == "metadata_ridge":
            fields = list(features.get("metadata", []))
            frame = self._models().reindex(model_ids)[fields].copy()
            return pd.get_dummies(frame.astype("string"), dummy_na=True, dtype=float)
        raise ValueError(f"Unsupported candidate strategy: {modality}")

    @staticmethod
    def _fit_predictor(strategy: str, alpha: float, k: int) -> Pipeline:
        model: Any = KNeighborsRegressor(n_neighbors=max(1, k), weights="distance") if strategy == "multiomics_knn" else Ridge(alpha=alpha)
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler()), ("model", model)])

    def _select_columns(self, x: pd.DataFrame, y: pd.Series, top_k: int) -> list[str]:
        scored: list[tuple[str, float]] = []
        for column in x.columns:
            pair = pd.concat([pd.to_numeric(x[column], errors="coerce"), y], axis=1).dropna()
            if len(pair) >= 8 and pair.iloc[:, 0].nunique() > 1:
                rho_raw, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                rho = float(cast(Any, rho_raw))
                if np.isfinite(rho):
                    scored.append((column, abs(float(rho))))
        return [column for column, _ in sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:max(1, top_k)]]

    def _cv_metrics(self, plan: dict[str, Any], x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        folds = min(int(self.config["research"].get("cv_folds", 5)), len(x))
        if folds < 3:
            return {"status": "INSUFFICIENT_DATA", "sample_size": len(x)}
        predictions = np.full(len(x), np.nan)
        splitter = KFold(n_splits=folds, shuffle=True, random_state=int(self.config["split"]["seed"]))
        x_array = x.reset_index(drop=True)
        y_array = y.reset_index(drop=True)
        top_k = int(plan.get("feature_selection", {}).get("top_k", x.shape[1]))
        for train_idx, holdout_idx in splitter.split(x_array):
            train_x = x_array.iloc[train_idx]
            selected = self._select_columns(train_x, y_array.iloc[train_idx], top_k) if plan.get("feature_selection", {}).get("nested", False) else list(x.columns)
            if not selected:
                selected = list(x.columns)
            fitted = self._fit_predictor(plan["strategy"], float(plan.get("alpha", 10.0)), int(plan.get("k", 10)))
            fitted.fit(train_x[selected], y_array.iloc[train_idx])
            predictions[holdout_idx] = fitted.predict(x_array.iloc[holdout_idx][selected])
        residual = y_array.to_numpy() - predictions
        return {"status": "OK", "sample_size": len(y_array), "folds": folds, "mae": float(mean_absolute_error(y_array, predictions)), "rmse": float(mean_squared_error(y_array, predictions) ** 0.5), "r2": float(r2_score(y_array, predictions)), "residual_std": float(np.std(residual, ddof=1)), "nested_feature_selection": bool(plan.get("feature_selection", {}).get("nested", False))}

    def _baseline_plan(self, target_gene: str) -> dict[str, Any]:
        return {"strategy": "multiomics_knn", "features_by_modality": {"expression": [target_gene], "cnv": [target_gene]}, "k": int(self.config["research"].get("knn_k", 10)), "alpha": 1.0, "feature_selection": {"nested": False, "top_k": 2}}

    def build_candidate(self, plan: dict[str, Any], target_gene: str, run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        canonical = dict(plan)
        canonical.setdefault("target_gene", target_gene)
        canonical.setdefault("feature_selection", {"nested": True, "top_k": min(20, sum(len(v) for v in canonical.get("features_by_modality", {}).values()))})
        canonical.setdefault("alpha", 10.0)
        canonical.setdefault("k", int(self.config["research"].get("knn_k", 10)))
        candidate_id = "candidate_" + sha256_json(canonical)[:18]
        if candidate_id in self._candidates:
            return {"status": "CACHED", **self._candidates[candidate_id]}
        x = self._feature_frame(canonical, list(split.train))
        y = self._effects(target_gene, split.train).reindex(x.index)
        usable = y.notna()
        x, y = x.loc[usable], y.loc[usable]
        if x.empty or x.shape[1] == 0:
            return {"status": "INSUFFICIENT_DATA", "warnings": ["No usable candidate features"]}
        nested = self._cv_metrics(canonical, x, y)
        stored = {"candidate_id": candidate_id, "target_gene": self.gene_symbol(target_gene), "plan": canonical, "train_cv": nested, "created_at": utc_now()}
        self._candidates[candidate_id] = stored
        self.store(run_id).json("build_prediction_candidate", stored, canonical, "TRAIN")
        return {"status": "OK", **stored}

    def _paired_bootstrap(self, baseline: np.ndarray, candidate: np.ndarray, y: np.ndarray) -> list[float]:
        rng = np.random.default_rng(int(self.config["split"]["seed"]))
        values = []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            values.append(float(np.mean(np.abs(y[idx] - baseline[idx])) - np.mean(np.abs(y[idx] - candidate[idx]))))
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    def evaluate_candidate(self, candidate_id: str, run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        if candidate_id not in self._candidates:
            import json
            for path in self.store(run_id).root.glob("*.json"):
                raw = json.loads(path.read_text(encoding="utf-8"))
                value = raw.get("value")
                if isinstance(value, dict) and value.get("candidate_id") == candidate_id:
                    self._candidates[candidate_id] = value
                    break
        if candidate_id not in self._candidates:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        submissions = self._validation_submissions.get(run_id, 0)
        limit = int(self.config["research"].get("max_validation_submissions", 4))
        if submissions >= limit:
            return {"status": "BUDGET_EXHAUSTED", "candidate_id": candidate_id, "previous_validation_submissions": submissions}
        stored = self._candidates[candidate_id]
        plan = stored["plan"]
        x_train = self._feature_frame(plan, list(split.train))
        x_validation = self._feature_frame(plan, list(split.validation))
        y_train = self._effects(stored["target_gene"], split.train).reindex(x_train.index)
        y_validation = self._effects(stored["target_gene"], split.validation).reindex(x_validation.index)
        usable = y_train.notna()
        selected = self._select_columns(x_train.loc[usable], y_train.loc[usable], int(plan.get("feature_selection", {}).get("top_k", x_train.shape[1]))) if plan.get("feature_selection", {}).get("nested", False) else list(x_train.columns)
        selected = selected or list(x_train.columns)
        model = self._fit_predictor(plan["strategy"], float(plan.get("alpha", 10.0)), int(plan.get("k", 10)))
        model.fit(x_train.loc[usable, selected], y_train.loc[usable])
        candidate_pred = model.predict(x_validation[selected])
        baseline = self._baseline_plan(stored["target_gene"])
        bx_train = self._feature_frame(baseline, list(split.train))
        bx_val = self._feature_frame(baseline, list(split.validation))
        base_model = self._fit_predictor("multiomics_knn", 1.0, int(baseline["k"]))
        base_model.fit(bx_train, y_train.reindex(bx_train.index))
        baseline_pred = base_model.predict(bx_val)
        valid = y_validation.notna().to_numpy()
        if valid.sum() < 3:
            return {"status": "INSUFFICIENT_DATA", "candidate_id": candidate_id, "previous_validation_submissions": submissions}
        yv = y_validation.to_numpy()[valid]
        candidate_pred = np.asarray(candidate_pred)[valid]
        baseline_pred = np.asarray(baseline_pred)[valid]
        paired = float(np.mean(np.abs(yv - baseline_pred)) - np.mean(np.abs(yv - candidate_pred)))
        interval = self._paired_bootstrap(baseline_pred, candidate_pred, yv)
        result = {"status": "OK", "candidate_id": candidate_id, "candidate_metrics": {"mae": float(mean_absolute_error(yv, candidate_pred)), "rmse": float(mean_squared_error(yv, candidate_pred) ** 0.5), "r2": float(r2_score(yv, candidate_pred))}, "baseline_metrics": {"mae": float(mean_absolute_error(yv, baseline_pred)), "rmse": float(mean_squared_error(yv, baseline_pred) ** 0.5), "r2": float(r2_score(yv, baseline_pred))}, "paired_improvement_mae": paired, "paired_improvement_ci": interval, "validation_sample_size": int(valid.sum()), "previous_validation_submissions": submissions + 1, "split_scope": "VALIDATION", "selected_features": selected}
        self._validation_submissions[run_id] = submissions + 1
        stored["validation"] = result
        self.store(run_id).json("evaluate_candidate_on_validation", result, {"candidate_id": candidate_id}, "VALIDATION")
        return result

    def compare_candidates(self, candidate_ids: list[str], target_gene: str, run_id: str, primary_metric: str = "mae") -> dict[str, Any]:
        values = [self._candidates[cid].get("validation", {}) for cid in candidate_ids if cid in self._candidates]
        ranked = sorted(values, key=lambda value: value.get("candidate_metrics", {}).get(primary_metric, float("inf")))
        return {"status": "OK" if ranked else "INSUFFICIENT_DATA", "primary_metric": primary_metric, "candidates": ranked, "best_candidate_id": ranked[0]["candidate_id"] if ranked else None}

    def freeze_plan(self, candidate_id: str, target_gene: str, run_id: str) -> dict[str, Any]:
        if candidate_id not in self._candidates:
            self.evaluate_candidate(candidate_id, run_id)
        if candidate_id not in self._candidates:
            raise KeyError(candidate_id)
        candidate = self._candidates[candidate_id]
        validation = candidate.get("validation")
        threshold = float(self.config["research"].get("minimum_validation_improvement", 0.01))
        robust = float(self.config["research"].get("minimum_robustness_fraction", 0.60))
        if validation and validation.get("paired_improvement_mae", -np.inf) >= threshold and validation.get("paired_improvement_ci", [-np.inf, np.inf])[0] > -threshold and validation.get("validation_sample_size", 0) >= 3:
            plan = candidate["plan"]
            status = "VALIDATED_PLAN_SELECTED"
            fallback = False
        else:
            plan = self._baseline_plan(target_gene)
            status = "BASELINE_FALLBACK_SELECTED"
            fallback = True
        frozen_id = "plan_" + sha256_json({"run_id": run_id, "target_gene": target_gene, "candidate_id": candidate_id, "plan": plan, "fallback": fallback})[:18]
        record = {"frozen_plan_id": frozen_id, "target_gene": self.gene_symbol(target_gene), "plan": plan, "selected_candidate_id": candidate_id, "fallback_used": fallback, "status": status, "frozen_at": utc_now(), "thresholds": {"minimum_validation_improvement": threshold, "minimum_robustness_fraction": robust}}
        self._plans[frozen_id] = record
        self.store(run_id).json("freeze_prediction_plan", record, {"candidate_id": candidate_id}, "TRAIN")
        return {"status": "OK", **record}

    def predict_test(self, frozen_plan_id: str, target_gene: str, test_model_id: str, run_id: str) -> dict[str, Any]:
        split = self.run(run_id)
        require_scope(split, [test_model_id], "TEST")
        if frozen_plan_id not in self._plans:
            raise PermissionError("Prediction requires a frozen plan in this run")
        record = self._plans[frozen_plan_id]
        plan = record["plan"]
        x_train = self._feature_frame(plan, list(split.train))
        x_test = self._feature_frame(plan, [test_model_id])
        y_train = self._effects(target_gene, split.train).reindex(x_train.index)
        usable = y_train.notna()
        selected = self._select_columns(x_train.loc[usable], y_train.loc[usable], int(plan.get("feature_selection", {}).get("top_k", x_train.shape[1]))) if plan.get("feature_selection", {}).get("nested", False) else list(x_train.columns)
        selected = selected or list(x_train.columns)
        model = self._fit_predictor(plan["strategy"], float(plan.get("alpha", 10.0)), int(plan.get("k", 10)))
        model.fit(x_train.loc[usable, selected], y_train.loc[usable])
        predicted = float(model.predict(x_test[selected])[0])
        cv = self._cv_metrics(plan, x_train.loc[usable, selected], y_train.loc[usable])
        residual = max(float(cv.get("residual_std", 0.2)), float(self.config["prediction"].get("minimum_interval_width", 0.05)))
        z = float(self.config["prediction"].get("interval_z", 1.96))
        interval = [predicted - z * residual, predicted + z * residual]
        category = "strong_effect" if predicted <= float(self.config["prediction"]["strong_effect_threshold"]) else "weak_effect" if predicted <= float(self.config["prediction"]["weak_effect_threshold"]) else "no_effect"
        artifact = self.store(run_id).json("predict_test_gene_effect", {"target_gene": self.gene_symbol(target_gene), "test_model_id": test_model_id, "frozen_plan_id": frozen_plan_id, "predicted_gene_effect": predicted, "uncertainty_interval": interval, "confidence": float(1.0 / (1.0 + residual)), "effect_category": category, "fallback_used": bool(record["fallback_used"]), "created_at": utc_now()}, {"frozen_plan_id": frozen_plan_id, "test_model_id": test_model_id}, "TEST")
        result = {"status": "OK", "target_gene": self.gene_symbol(target_gene), "test_model_id": test_model_id, "frozen_plan_id": frozen_plan_id, "predicted_gene_effect": predicted, "uncertainty_interval": interval, "confidence": float(1.0 / (1.0 + residual)), "effect_category": category, "fallback_used": bool(record["fallback_used"]), "prediction_artifact_id": artifact, "cv_metrics": cv}
        self._predictions[artifact] = result
        return result

    def evaluate_final(self, prediction_artifact_id: str, evaluation_token: str, run_id: str) -> dict[str, Any]:
        if not evaluation_token.startswith("EVAL-"):
            raise PermissionError("A separate evaluation token is required")
        if prediction_artifact_id not in self._predictions:
            raise PermissionError("Prediction must be persisted before evaluation")
        result = self._predictions[prediction_artifact_id]
        actual = self._effects(result["target_gene"], [result["test_model_id"]]).iloc[0]
        prediction = float(result["predicted_gene_effect"])
        return {"status": "OK", "prediction_artifact_id": prediction_artifact_id, "actual_gene_effect": float(actual), "absolute_error": abs(float(actual) - prediction), "squared_error": (float(actual) - prediction) ** 2, "evaluation_scope": "EVALUATION"}

