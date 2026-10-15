from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any


def evidence_id(hypothesis_id: str, tool: str, artifact_id: str | None, estimate: Any) -> str:
    raw = repr((hypothesis_id, tool, artifact_id, estimate)).encode()
    return "evidence_" + hashlib.sha256(raw).hexdigest()[:16]


def register(hypothesis_id: str, source_tool: str, result: dict[str, Any]) -> dict[str, Any]:
    artifact = result.get("artifact_id")
    estimate = result.get("estimate", result.get("paired_improvement_mae"))
    return {"evidence_id": evidence_id(hypothesis_id, source_tool, artifact, estimate), "hypothesis_id": hypothesis_id, "source_tool": source_tool, "tool_call_id": result.get("tool_call_id"), "artifact_id": artifact, "split_scope": result.get("split_scope", "TRAIN"), "model_population": result.get("sample_size", result.get("validation_sample_size")), "features": result.get("features", []), "statistical_method": result.get("method"), "estimate": estimate, "effect_size": result.get("effect_size"), "p_value": result.get("p_value"), "adjusted_p_value": result.get("adjusted_p_value"), "confidence_interval": result.get("confidence_interval", result.get("paired_improvement_ci")), "support_direction": "SUPPORTING" if result.get("status") == "OK" else "INCONCLUSIVE", "quality_flags": result.get("warnings", []), "limitations": result.get("limitations", []), "provenance": result.get("provenance", {}), "timestamp": datetime.now(timezone.utc).isoformat()}
