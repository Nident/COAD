from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .manifests import load_config, source_manifest, utc_now, write_json


def prepare_file(source: Path, target: Path, chunksize: int = 512) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    writer: pq.ParquetWriter | None = None
    rows = 0
    columns: list[str] = []
    try:
        for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
            if "Unnamed: 0" in chunk.columns and chunk.columns[0] == "Unnamed: 0":
                chunk = chunk.rename(columns={"Unnamed: 0": "ModelID"}) if "ModelID" not in chunk.columns else chunk.drop(columns=["Unnamed: 0"])
            # Mutation annotations contain columns whose inferred dtype changes
            # across chunks (for example bool/string missingness).  They are
            # categorical annotations, so a stable string schema is lossless
            # for the supported mutation representations.
            if "Mutation" in source.name or source.name == "Model.csv":
                chunk = chunk.astype("string")
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                columns = list(chunk.columns)
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return {"path": str(target), "rows": rows, "columns": columns}


def prepare(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data = config["data"]
    root = (config_path.parent / data["root"]).resolve()
    cache = (config_path.parent / data["cache_dir"]).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    sources = source_manifest(config, config_path)
    outputs = {}
    for key in ("crispr", "expression", "mutations", "cnv", "metadata"):
        outputs[key] = prepare_file(root / data[key], cache / f"{key}.parquet")
    manifest = {"cache_version": data.get("cache_version", "1"), "prepared_at": utc_now(), "sources": sources, "outputs": outputs}
    write_json(cache / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible Parquet cache for DepMap data")
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    args = parser.parse_args()
    result = prepare(args.config)
    print(json.dumps({"manifest": str((args.config.parent / "../cache/data_manifest.json").resolve()), "datasets": list(result["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
