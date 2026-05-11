# Research Journal

## 2026-05-11 Project Scaffold

Created the initial compact research workspace around the local NeuroGolf dataset. The immediate priority is infrastructure: validate tasks, summarize the dataset, then build a repeatable loop for candidate ONNX generation and verification.

## Operating Loop

1. Select a small batch of tasks.
2. Infer the transformation family.
3. Propose the smallest plausible ONNX architecture.
4. Generate a candidate model in `models/candidates/`.
5. Validate against `train`, `test`, and `arc-gen`.
6. Move passing models to `models/verified/`.
7. Record structured results in `experiments/experiments.jsonl`.
8. Update hypotheses, failures, and successful patterns.

## Roadmap

### Phase 1: Infrastructure

- Validate all task files.
- Build dataset summaries and task taxonomy.
- Wrap official validation locally.
- Create repeatable model generation and scoring commands.

### Phase 2: Baseline Families

- Identity and color remapping.
- Single-layer local convolutions.
- Static geometric transforms.
- Crop, pad, tile, and scale-like transformations.

### Phase 3: Search and Compression

- Search over sparse operator families.
- Record exact generalization failures.
- Compress passing candidates.
- Compare parameter count and memory cost.

### Phase 4: Submission Assembly

- Keep only verified ONNX files.
- Build `submission.zip`.
- Track public/private score deltas.

## Meta Reviews

No meta-review recorded yet.

