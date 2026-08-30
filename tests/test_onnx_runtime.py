"""Tests for tekmyra.onnx_runtime."""

from __future__ import annotations

import numpy as np
import pytest

from tekmyra.onnx_runtime import OnnxSessionManager


def _make_mul2_onnx(path):
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 2])
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    node = helper.make_node("Mul", ["X", "two"], ["Y"])
    graph = helper.make_graph([node], "mul2", [X], [Y], [two])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    onnx.save(model, str(path))


def test_manager_loads_session_and_runs(tmp_path):
    path = tmp_path / "mul2.onnx"
    _make_mul2_onnx(path)

    mgr = OnnxSessionManager(path, config=None)
    assert mgr.loaded
    assert "X" in mgr.input_names
    assert "Y" in mgr.output_names
    out = mgr.run({"X": np.array([[1.0, 3.0]], dtype=np.float32)})
    assert np.allclose(out["Y"], [[2.0, 6.0]])


def test_manager_io_binding_can_be_disabled(tmp_path):
    from tekmyra.onnx_runtime import OrtConfig

    path = tmp_path / "mul2.onnx"
    _make_mul2_onnx(path)

    cfg = OrtConfig(model_path=path, use_io_binding=False, warmup_iterations=1)
    mgr = OnnxSessionManager(path, config=cfg)
    assert mgr.loaded
    out = mgr.run({"X": np.array([[2.0, 5.0]], dtype=np.float32)})
    assert np.allclose(out["Y"], [[4.0, 10.0]])


def test_manager_deterministic_fallback_on_bad_path():
    mgr = OnnxSessionManager("/nonexistent/model.onnx")
    assert not mgr.loaded
    with pytest.raises(RuntimeError, match="ONNX session not loaded"):
        mgr.run({"X": np.array([[1.0]])})
