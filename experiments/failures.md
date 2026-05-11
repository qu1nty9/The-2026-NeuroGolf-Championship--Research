# Failure Patterns

No model failures recorded yet.

Expected failure categories to track:

- Correct on `train` but fails on `arc-gen`.
- Uses too many parameters for a simple transformation.
- Produces invalid one-hot output.
- Relies on oversized public examples that official local validation skips.
- Solves shape but not color semantics.

