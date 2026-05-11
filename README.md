# NeuroGolf 2026 Autonomous Research Agent

This repository is a compact research workspace for **The 2026 NeuroGolf Championship**. The goal is to study whether an autonomous AI research agent can independently discover, build, validate, and compress small neural programs for ARC-AGI image transformation tasks.

The project is not only about leaderboard score. It is also an experiment in autonomous research behavior: hypothesis formation, architecture search, failure analysis, persistent memory, and iterative improvement of compact ONNX solutions.

## Competition Task

In NeuroGolf, each of the 400 ARC-AGI tasks must be solved by a separate ONNX network. Each network receives an input grid encoded as a tensor:

```text
1 x 10 x 30 x 30
```

The 10 channels represent colors `0..9`. The network must output another one-hot tensor from which the target ARC grid can be reconstructed exactly.

Each submitted network must:

- be functionally correct on `train`, `test`, `arc-gen`, and hidden evaluation examples;
- use as few parameters as possible;
- use as little memory as possible;
- be a valid ONNX model with static tensor shapes;
- avoid banned operators: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, `Compress`;
- stay below the `1.44MB` per-model file size limit.

The score for a correct network depends on its cost:

```text
cost = params + memory_bytes
score = max(1, 25 - ln(cost))
```

The central research question is:

> What is the smallest computational structure required to express each abstract ARC transformation?

## Research Hypothesis

Many ARC tasks may not require large trained neural networks. They may be solvable by compact, interpretable neural programs assembled from a small number of static ONNX operators.

This project studies:

- long-horizon autonomous AI research behavior;
- architecture discovery;
- neural program synthesis;
- model compression;
- generalization from `train/test` to `arc-gen` and hidden evaluation;
- recurring failure modes of autonomous agents;
- the relationship between an ARC rule and its minimum parameter count.

## Example Research Task: `task001`

The first running example is [`data/task001.json`](data/task001.json).

This task is a useful first benchmark because:

- every input is `3x3`;
- every output is `9x9`;
- it has 5 `train` examples, 1 `test` example, and 262 `arc-gen` examples;
- the rule is visually simple, but expressing it as a compact neural network is nontrivial.

Observed rule:

> The input `3x3` pattern acts as a mask. Each non-zero input cell expands into a copy of the full input pattern, while each zero cell expands into an empty `3x3` block. The result is a `9x9` grid.

In other words, the task resembles a self-tiling or Kronecker-style expansion:

```text
input 3x3 -> output 9x9
non-zero cell -> copy of input pattern
zero cell     -> 3x3 zero block
```

This makes `task001` a good first research target: it forces the system to search for a small compositional architecture rather than a simple color classifier or brute-force memorizer.

## Compact Project Layout

```text
.
├── data/                  # local raw task*.json + official neurogolf_utils, not tracked by git
├── src/neurogolf/         # reusable Python package and CLI
├── experiments/
│   ├── journal.md         # narrative research journal and roadmap
│   ├── experiments.jsonl  # machine-readable experiment events
│   ├── hypotheses.md      # active research hypotheses
│   ├── failures.md        # recurring failure patterns
│   ├── successes.md       # successful architecture patterns
│   └── reports/           # dataset summaries and validation reports
├── models/
│   ├── candidates/        # generated ONNX candidates
│   └── verified/          # public-validation-passing ONNX models
├── notebooks/             # exploratory notebooks only
├── submissions/           # generated submission.zip files
├── tests/                 # infrastructure tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

`data/` is treated as the immutable input layer. Project code may read the raw task files and official utilities, but should not modify them.

The raw competition dataset is intentionally not tracked in git. To reproduce the local workspace, download the NeuroGolf competition files from Kaggle and place them under `data/` so that paths look like:

```text
data/task001.json
data/task002.json
...
data/task400.json
data/neurogolf_utils/neurogolf_utils.py
```

## Current Dataset Snapshot

The latest dataset summary is stored in [`experiments/reports/dataset_summary.json`](experiments/reports/dataset_summary.json).

Current state:

- `400` task files;
- `400` valid tasks;
- `101718` total examples;
- `1302` train examples;
- `416` test examples;
- `100000` arc-gen examples;
- `138` tasks change input/output size;
- `371` public examples exceed `30x30` and are skipped by the official local verifier.

Important note: oversized `arc-gen` examples are part of the raw dataset, but the official validation utility skips examples that cannot be represented within the `30x30` tensor format.

## Setup

Create a local environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Commands can be run directly from the repository root with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m neurogolf dataset-summary --data-dir data --write experiments/reports/dataset_summary.json
PYTHONPATH=src python -m neurogolf inspect-task 001 --data-dir data
PYTHONPATH=src python -m neurogolf doctor
PYTHONPATH=src python -m neurogolf build-task001-baseline
PYTHONPATH=src python -m unittest discover -s tests
```

`doctor` checks the dependencies needed for official ONNX validation: `onnx`, `onnxruntime`, `onnx-tool`, `matplotlib`, and `ipython`.

## Research Workflow

Every meaningful experiment should leave two traces:

1. A structured event in `experiments/experiments.jsonl`.
2. A short interpretation in `experiments/journal.md`.

Basic loop:

1. Select a task or a small cluster of tasks.
2. Describe the hypothesized transformation.
3. Choose the smallest plausible architecture family.
4. Generate an ONNX candidate in `models/candidates/`.
5. Validate it on `train`, `test`, and `arc-gen`.
6. Move passing models to `models/verified/`.
7. Record parameters, memory, estimated score, and failure details.
8. Update `hypotheses.md`, `failures.md`, or `successes.md`.

Research priorities:

```text
1. correctness
2. generalization
3. compression
4. simplicity
5. architectural elegance
```

## Architecture Families To Explore

Initial families:

- identity networks and simple color remapping;
- single-layer sparse convolution;
- local neighborhood rules;
- crop, pad, tile, and scale transforms;
- mask extraction and broadcasting;
- static geometric transforms;
- compositions of small operators instead of dense memorization.

For `task001`, the first useful milestone is a compact ONNX candidate that implements the self-tiling rule without memorizing all `arc-gen` examples.

Current baseline command:

```bash
PYTHONPATH=src python -m neurogolf build-task001-baseline
```

This writes `models/candidates/task001.onnx`, validates it with the official local utilities, stores a report in `experiments/reports/task001_baseline_validation.json`, and appends a structured event to `experiments/experiments.jsonl`.

## GitHub Workflow

The repository is prepared for GitHub:

- generated ONNX models and submission archives are ignored through `.gitignore`;
- research state in `experiments/` can be committed to preserve the history of hypotheses and results;
- raw `data/` is intentionally ignored and should be downloaded locally from Kaggle.

GitHub Actions can be added later. Creating workflow files requires a GitHub token with `workflow` scope.

Initial publication:

```bash
git init
git add .
git commit -m "Initialize compact NeuroGolf research workspace"
gh repo create neurogolf-2026-agent --private --source=. --push
```

Suggested branch style:

```text
infra/...
experiment/task001-...
model-family/...
submission/...
```

Suggested commit style:

```text
infra: compact project architecture
data: refresh dataset summary
experiment: analyze task001 self-tiling rule
model: add task001 sparse baseline
eval: record task001 validation failure
```

## Status

Project scaffold is ready, and the first `task001` baseline passes local public validation.

Current `task001` baseline:

- ARC-AGI: 6 pass, 0 fail;
- ARC-GEN: 262 pass, 0 fail;
- parameters: 5;
- memory: 14328 bytes;
- estimated score: 15.430.

Next milestone: compress the `task001` solution and generalize the pipeline to additional task families.
