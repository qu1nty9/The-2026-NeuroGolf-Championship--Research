from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from .config import GRID_SHAPE, PATHS


IR_VERSION = 10
OPSET_IMPORTS = [helper.make_opsetid("", 10)]
FLOAT = TensorProto.FLOAT
INT64 = TensorProto.INT64


def _tensor(name: str, data_type: int, values: np.ndarray) -> onnx.TensorProto:
    return helper.make_tensor(name, data_type, list(values.shape), values.flatten().tolist())


def build_task001_self_tiling_model() -> onnx.ModelProto:
    """Build a compact baseline for task001's 3x3 -> 9x9 self-tiling rule."""

    nodes = [
        helper.make_node(
            "Slice",
            ["input", "spatial_starts", "spatial_ends", "spatial_axes"],
            ["pattern"],
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
            ["pattern", "channel_starts", "channel_ends", "channel_axes"],
            ["nonzero_channels"],
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
            "Reshape",
            ["mask_3x3", "mask_block_shape"],
            ["mask_blocked"],
            name="reshape_mask_for_repeat",
        ),
        helper.make_node(
            "Tile",
            ["mask_blocked", "mask_block_repeats"],
            ["mask_repeated_blocked"],
            name="repeat_mask_blocks",
        ),
        helper.make_node(
            "Reshape",
            ["mask_repeated_blocked", "mask_9x9_shape"],
            ["mask_9x9"],
            name="reshape_mask_to_9x9",
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
            "Mul",
            ["inactive_mask_9x9", "zero_channel_selector"],
            ["inactive_zero_blocks"],
            name="make_inactive_blocks_color_zero",
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

    selector = np.zeros((1, 10, 1, 1), dtype=np.float32)
    selector[:, 0, :, :] = 1.0

    initializers = [
        _tensor("spatial_starts", INT64, np.array([0, 0], dtype=np.int64)),
        _tensor("spatial_ends", INT64, np.array([3, 3], dtype=np.int64)),
        _tensor("spatial_axes", INT64, np.array([2, 3], dtype=np.int64)),
        _tensor("pattern_repeats", INT64, np.array([1, 1, 3, 3], dtype=np.int64)),
        _tensor("channel_starts", INT64, np.array([1], dtype=np.int64)),
        _tensor("channel_ends", INT64, np.array([10], dtype=np.int64)),
        _tensor("channel_axes", INT64, np.array([1], dtype=np.int64)),
        _tensor("mask_block_shape", INT64, np.array([1, 1, 3, 1, 3, 1], dtype=np.int64)),
        _tensor("mask_block_repeats", INT64, np.array([1, 1, 1, 3, 1, 3], dtype=np.int64)),
        _tensor("mask_9x9_shape", INT64, np.array([1, 1, 9, 9], dtype=np.int64)),
        _tensor("one_scalar", FLOAT, np.array([1.0], dtype=np.float32)),
        _tensor("zero_channel_selector", FLOAT, selector),
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

