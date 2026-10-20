# Context-Specific Cancer Dependency Discovery

This repository explores whether a cancer cell line's molecular state can
predict its response to a gene knockout. The main output is a **continuous
DepMap CRISPR Gene Effect** for a selected target gene and cell line:

- more negative values indicate stronger dependency;
- values closer to zero indicate weaker or no dependency.

The project combines RNA expression, somatic mutations, copy-number variation
(CNV), and cell-line metadata. Its architecture evolved from exploratory
notebooks to a leakage-controlled autonomous research agent.

## Pipeline overview

```text
Target gene + held-out ModelID
        -> inspect training multi-omics data
        -> discover predictive relationships
        -> build and validate candidate predictors
        -> freeze the selected prediction plan
        -> predict continuous Gene Effect
        -> open the hidden label only for final evaluation
```

The test model's real Gene Effect must not participate in feature selection,
retrieval, model fitting, validation, or prediction.

## Experiments

| Stage | Purpose | Where to start |
|---|---|---|
| [Pilot](experiments/pilot/) | Initial data inspection, KRAS analysis, and classical ML baselines. | Notebooks in the directory |
| [1. Context-Specific Dependency Discovery](experiments/1_Context-Specific_Dependency_Discovery/) | Single-LLM prediction from donor, expression, mutation, and CRISPR tables. | [Experiment README](experiments/1_Context-Specific_Dependency_Discovery/README.md) |
| [2. Multi-LLM](experiments/2_CSDD_MULTI_LLM/) | Separate modality agents produce hypotheses; a judge combines them. | [Experiment README](experiments/2_CSDD_MULTI_LLM/README.md) |
| [3. Multi-Omics RAG](experiments/3_CSDD_RAG/) | Expression-first retrieval, mutation/CNV/metadata reranking, KNN baseline, and LLM correction. | [Experiment README](experiments/3_CSDD_RAG/README.md) |
| [4. AgentLoop + MCP](experiments/4_AgentLoop_MCP/) | Current pipeline: a LangGraph agent selects FastMCP statistical tools, tests relationships, validates deterministic predictors, and predicts Gene Effect. | [Experiment README](experiments/4_AgentLoop_MCP/README.md) |

Use **Experiment 4** for the current implementation. Earlier stages are kept to
show how the research architecture developed and to support comparisons.

## Main data

Raw CCLE/DepMap tables are expected in [`data/`](data/):

| File | Content |
|---|---|
| `CRISPRGeneEffect.csv` | Observed gene-dependency values used as prediction targets |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | RNA expression features |
| `OmicsSomaticMutations.csv` | Somatic mutation records |
| `OmicsCNGeneWGS.csv` | Gene-level copy-number values |
| `Model.csv` | Cell-line, disease, lineage, and donor metadata |

Generated caches, prompts, trajectories, and prediction results remain inside
their corresponding experiment directories.

## Quick start: current pipeline

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd experiments/4_AgentLoop_MCP
```

Prepare the columnar data cache:

```bash
python3 -m data.prepare --config config/settings.yaml
```

Predict one held-out cell line:

```bash
python3 main.py \
  --gene 'CRKL (1399)' \
  --test-model-id ACH-000475 \
  --config config/settings.yaml
```

Run the configured batch benchmark:

```bash
python3 batch_runner.py \
  --batch-config config/batch_settings.yaml \
  --config config/settings.yaml
```

Configuration lives in
[`experiments/4_AgentLoop_MCP/config/`](experiments/4_AgentLoop_MCP/config/).
Store API credentials only in the ignored `config/secrets.yaml`, using
`config/secrets.yaml.example` as a template.

## What to inspect after a run

Inside [`experiments/4_AgentLoop_MCP/`](experiments/4_AgentLoop_MCP/):

- `results/` — final prediction reports in JSON and Markdown;
- `artifacts/<run_id>/` — split manifest, tables, model outputs, and prediction artifacts;
- `trajectories/<run_id>.jsonl` — ordered agent decisions and MCP tool calls;
- `batch_results/` — resumable batch predictions and aggregate metrics;
- `cache/data_manifest.json` — source schemas and cache provenance.

The central implementation is split into `agent/` for LangGraph orchestration,
`mcp_server/` for controlled data and statistical tools, `data/` for loading and
split enforcement, and `evaluation/` for post-prediction label access.

## Tests

```bash
cd experiments/4_AgentLoop_MCP
python3 -m unittest discover -s tests -v
pyright
```

The default tests use deterministic fixtures and do not require a paid LLM API.
Results describe statistical associations in cancer cell-line models and should
not be interpreted as clinical recommendations or causal biological proof.
