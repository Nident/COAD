from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_gene: str
    test_model_id: str
    split_manifest_id: str
    prediction_status: str
    selected_or_fallback_plan_id: str
    predicted_gene_effect: float
    uncertainty_interval: list[float] = Field(min_length=2, max_length=2)
    confidence: float
    effect_category: str
    fallback_used: bool
    validation_metrics: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    contradictory_hypotheses: list[dict[str, Any]]
    limitations: list[str]
    termination_reason: str
    evidence_ids: list[str]
    artifact_ids: list[str]
    trajectory_path: str


def save_report(report_data: dict[str, Any], json_path: Path, markdown_path: Path) -> PredictionReport:
    report = PredictionReport.model_validate(report_data)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown = [f"# DepMap Gene Effect prediction: {report.target_gene}", "", f"- Test ModelID: `{report.test_model_id}`", f"- Status: `{report.prediction_status}`", f"- Plan: `{report.selected_or_fallback_plan_id}`", f"- Predicted Gene Effect: `{report.predicted_gene_effect:.6f}`", f"- Uncertainty interval: `[{report.uncertainty_interval[0]:.6f}, {report.uncertainty_interval[1]:.6f}]`", f"- Category: `{report.effect_category}`", f"- Confidence: `{report.confidence:.3f}`", f"- Fallback used: `{report.fallback_used}`", "", "## Findings", ""]
    markdown.extend(f"- {item.get('claim', item.get('source_tool', 'finding'))} (evidence: {item.get('evidence_id', 'none')})" for item in report.findings)
    markdown.extend(["", "## Limitations", ""])
    markdown.extend(f"- {item}" for item in report.limitations)
    markdown.extend(["", f"Termination: `{report.termination_reason}`", f"", f"Trajectory: `{report.trajectory_path}`"])
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report

