from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from .config import GRID_SHAPE, PATHS


IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 7)]
FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64


def _tensor(name: str, data_type: int, values: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(name, data_type, list(values.shape), values.flatten().tolist())


def build_task001_self_tiling_model() -> onnx.ModelProto:
    """Build a compact baseline for task001's 3x3 -> 9x9 self-tiling rule.

    The model uses opset 7 so that Slice and Upsample constants can live in
    operator attributes instead of parameter-counted initializers.
    """

    nodes = [
        helper.make_node(
            "Slice",
            ["input"],
            ["pattern"],
            axes=[2, 3],
            starts=[0, 0],
            ends=[3, 3],
            name="slice_pattern_3x3",
        ),
        helper.make_node(
            "Tile",
            ["pattern", "pattern_repeats"],
            ["pattern_tiled"],
            name="tile_pattern_to_9x9",
        ),
        helper.make_node(
            "Slice",
            ["pattern"],
            ["nonzero_channels"],
            axes=[1],
            starts=[1],
            ends=[10],
            name="slice_nonzero_channels",
        ),
        helper.make_node(
            "ReduceSum",
            ["nonzero_channels"],
            ["mask_3x3"],
            axes=[1],
            keepdims=1,
            name="reduce_nonzero_mask",
        ),
        helper.make_node(
            "Upsample",
            ["mask_3x3"],
            ["mask_9x9"],
            mode="nearest",
            scales=[1.0, 1.0, 3.0, 3.0],
            name="upsample_mask_blocks",
        ),
        helper.make_node(
            "Mul",
            ["pattern_tiled", "mask_9x9"],
            ["active_blocks"],
            name="apply_active_mask",
        ),
        helper.make_node(
            "Sub",
            ["one_scalar", "mask_9x9"],
            ["inactive_mask_9x9"],
            name="invert_mask",
        ),
        helper.make_node(
            "Pad",
            ["inactive_mask_9x9"],
            ["inactive_zero_blocks"],
            pads=[0, 0, 0, 0, 0, 9, 0, 0],
            mode="constant",
            value=0.0,
            name="pad_inactive_mask_to_zero_channel",
        ),
        helper.make_node(
            "Add",
            ["active_blocks", "inactive_zero_blocks"],
            ["top_left_9x9_output"],
            name="merge_blocks",
        ),
        helper.make_node(
            "Pad",
            ["top_left_9x9_output"],
            ["output"],
            pads=[0, 0, 0, 0, 0, 0, 21, 21],
            mode="constant",
            value=0.0,
            name="pad_to_competition_shape",
        ),
    ]

    initializers = [
        _tensor("pattern_repeats", INT64, np.array([1, 1, 3, 3], dtype=np.int64)),
        _tensor("one_scalar", FLOAT, np.array([1.0], dtype=np.float32)),
    ]

    graph = helper.make_graph(
        nodes,
        "task001_self_tiling_baseline",
        [helper.make_tensor_value_info("input", FLOAT, GRID_SHAPE)],
        [helper.make_tensor_value_info("output", FLOAT, GRID_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=OPSET_IMPORTS)
    onnx.checker.check_model(model, full_check=True)
    return model


def save_task001_self_tiling_model(path: str | Path = PATHS.candidate_models / "task001.onnx") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(build_task001_self_tiling_model(), target)
    return target
