# Hypotheses

- Many tasks may be solvable by compact hand-structured operator families rather than learned dense networks.
- Task families should be identified before architecture search to avoid wasting parameters.
- A small set of reusable architecture families may cover a meaningful subset of the 400 tasks: identity/color maps, local convolutions, geometric transforms, crop/pad/tile/scale, and mask extraction.
- Mask expansion plus pattern tiling is a promising reusable family for tasks that use an input grid as both content and control structure.
- In NeuroGolf scoring, after parameter count is compressed below roughly tens of constants, intermediate tensor memory dominates; future compression should focus on reducing materialized tensors.
