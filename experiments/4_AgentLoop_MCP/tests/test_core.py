from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from data.access_policy import AccessDenied
from data.repository import DataRepository
from mcp_server.server import create_server
from fastmcp import Client
from agent.graph import MCPResearchAgent


class CoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data = root / "data"; data.mkdir()
        models = [f"M{i:02d}" for i in range(1, 16)]
        pd.DataFrame({"ModelID": models, "ACO2 (26)": [-0.1 - i / 20 for i in range(15)], "B (2)": [i / 10 for i in range(15)]}).to_csv(data / "CRISPRGeneEffect.csv", index=False)
        rows = [{"ModelID": m, "IsDefaultEntryForModel": "Yes", "ACO2 (26)": i / 10, "B (2)": i / 20} for i, m in enumerate(models)]
        pd.DataFrame(rows).to_csv(data / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", index=False)
        pd.DataFrame(rows).to_csv(data / "OmicsCNGeneWGS.csv", index=False)
        pd.DataFrame({"ModelID": models, "PatientID": [f"P{i}" for i in range(15)], "OncotreeLineage": ["Liver" if i % 2 else "Lung" for i in range(15)]}).to_csv(data / "Model.csv", index=False)
        pd.DataFrame({"ModelID": ["M01", "M01", "M02"], "IsDefaultEntryForModel": ["Yes", "Yes", "Yes"], "HugoSymbol": ["ACO2", "ACO2", "B"], "EntrezGeneID": [26, 26, 2], "ProteinChange": ["p.A", "p.A", "p.B"]}).to_csv(data / "OmicsSomaticMutations.csv", index=False)
        config = {"data": {"root": "data", "crispr": "CRISPRGeneEffect.csv", "expression": "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv", "mutations": "OmicsSomaticMutations.csv", "cnv": "OmicsCNGeneWGS.csv", "metadata": "Model.csv", "cache_dir": "cache", "cache_version": "test"}, "split": {"seed": 1, "validation_fraction": 0.2, "patient_grouped": True}, "research": {"cv_folds": 3, "knn_k": 2, "max_validation_submissions": 2, "minimum_effect_size": 0.15, "minimum_validation_improvement": 0.01, "minimum_robustness_fraction": 0.6}, "prediction": {"strong_effect_threshold": -1.0, "weak_effect_threshold": -0.5, "interval_z": 1.96, "minimum_interval_width": 0.05}, "artifacts": {"root": "artifacts", "trajectories": "trajectories", "results": "results"}}
        self.config_path = root / "config.yaml"; self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        self.repo = DataRepository(self.config_path)
        self.run_id = "run_test"
        self.repo.initialize_run("ACO2", "M15", self.run_id)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_split_boundaries_and_default_rows(self) -> None:
        split = self.repo.run(self.run_id)
        with self.assertRaises(AccessDenied):
            self.repo.train_effect("ACO2", ["M15"], self.run_id, 0, 20)
        profile = self.repo.test_profile("ACO2", "M15", ["expression", "mutation", "cnv"], self.run_id)
        self.assertNotIn("GeneEffect", profile["profile"])
        self.assertIn("protected_fields", profile)
        mutation = self.repo.get_mutations(["ACO2"], ["M01"], "gene", self.run_id, 0, 10)
        self.assertEqual(mutation["rows"][0]["ACO2"], 1.0)

    def test_candidate_freeze_and_prediction_gate(self) -> None:
        plan = {"strategy": "expression_ridge", "features_by_modality": {"expression": ["ACO2"]}, "alpha": 1.0, "feature_selection": {"nested": True, "top_k": 1}}
        candidate = self.repo.build_candidate(plan, "ACO2", self.run_id)
        self.assertEqual(candidate["status"], "OK")
        self.assertEqual(candidate["candidate_id"], self.repo.build_candidate(plan, "ACO2", self.run_id)["candidate_id"])
        with self.assertRaises(PermissionError):
            self.repo.predict_test("not-frozen", "ACO2", "M15", self.run_id)
        self.repo.evaluate_candidate(candidate["candidate_id"], self.run_id)
        frozen = self.repo.freeze_plan(candidate["candidate_id"], "ACO2", self.run_id)
        prediction = self.repo.predict_test(frozen["frozen_plan_id"], "ACO2", "M15", self.run_id)
        self.assertIsInstance(prediction["predicted_gene_effect"], float)

    def test_mcp_discovery(self) -> None:
        async def check() -> None:
            async with Client(create_server(self.config_path)) as client:
                names = {tool.name for tool in await client.list_tools()}
                self.assertIn("initialize_run", names)
                self.assertIn("predict_test_gene_effect", names)
        asyncio.run(check())

    def test_dynamic_policy_changes_tool_sequence(self) -> None:
        agent = MCPResearchAgent(self.config_path, "run_policy", "ACO2", "M15")
        state = agent.initial_state()
        state["results"] = {"initialize_run": {}, "get_target_gene_summary": {}, "get_test_model_profile": {}, "screen_expression_dependency_associations": {"rows": [{"feature": "ACO2 (26)", "estimate": 0.4}]}}
        expression_action = agent._policy_action(state)
        state["results"]["screen_expression_dependency_associations"] = {"rows": [{"feature": "ACO2 (26)", "estimate": 0.02}]}
        cnv_action = agent._policy_action(state)
        self.assertEqual(expression_action["tool"], "build_prediction_candidate")
        self.assertEqual(cnv_action["tool"], "screen_cnv_dependency_associations")


if __name__ == "__main__":
    unittest.main()
