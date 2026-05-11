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

## 2026-05-11 Task001 Self-Tiling Baseline

Built the first complete model-generation and validation loop for `task001`. The baseline implements the observed self-tiling rule: each non-zero input cell activates a copy of the original `3x3` pattern, while each zero input cell activates a `3x3` color-zero block. The generated ONNX model passes all available public examples for the task.

Validation summary:

- ARC-AGI examples: 6 pass, 0 fail.
- ARC-GEN examples: 262 pass, 0 fail.
- Parameters: 40.
- Memory: 14688 bytes.
- Estimated score: 15.402.

The model is a correctness baseline rather than a compression endpoint. The next research question is whether the same transformation can be expressed with fewer constants and less intermediate memory.

## 2026-05-11 Task001 Compression Pass

Compressed the `task001` baseline by moving Slice and Upsample constants into opset-7 attributes, replacing the explicit color-zero selector with channel padding, and using `Upsample` instead of block-repeat reshape/tile operations for the mask. The model remains correct on all public examples.

Compression result:

- ARC-AGI examples: 6 pass, 0 fail.
- ARC-GEN examples: 262 pass, 0 fail.
- Parameters: 5, down from 40.
- Memory: 14328 bytes, down from 14688.
- Estimated score: 15.430, up from 15.402.

Remaining cost is dominated by intermediate tensors, not parameters. The next compression target is reducing the `pattern_tiled`, `active_blocks`, and inactive color-zero intermediate memory.
