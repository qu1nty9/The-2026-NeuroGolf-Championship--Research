# Successful Patterns

- `task001` self-tiling can be represented as: slice `3x3` pattern, tile pattern to `9x9`, derive a non-zero mask from channels `1..9`, repeat mask into `3x3` blocks, merge active pattern blocks with inactive color-zero blocks, then pad to `30x30`.
