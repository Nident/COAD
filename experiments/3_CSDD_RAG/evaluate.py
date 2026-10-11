import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def main() -> None:
    runs_dir = Path(__file__).parent / "runs"
    rows = [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in runs_dir.glob("*/evaluation.json")
    ]
    table = pd.DataFrame(rows)
    predicted = table["PredictedGeneEffect"].to_numpy(dtype=float)
    actual = table["ActualGeneEffect"].to_numpy(dtype=float)
    metrics = {
        "n": len(table),
        "MAE": float(np.mean(np.abs(predicted - actual))),
        "RMSE": float(np.sqrt(np.mean(np.square(predicted - actual)))),
        "classification_accuracy": float(
            np.mean(table["PredictedCategory"] == table["ActualCategory"])
        ),
    }
    if len(table) >= 2:
        predicted_ranks = pd.Series(predicted).rank().to_numpy(dtype=float)
        actual_ranks = pd.Series(actual).rank().to_numpy(dtype=float)
        metrics["PearsonCorrelation"] = float(
            np.corrcoef(predicted, actual)[0, 1]
        )
        metrics["SpearmanCorrelation"] = float(
            np.corrcoef(predicted_ranks, actual_ranks)[0, 1]
        )
    else:
        metrics["PearsonCorrelation"] = None
        metrics["SpearmanCorrelation"] = None
    (runs_dir / "evaluation_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
