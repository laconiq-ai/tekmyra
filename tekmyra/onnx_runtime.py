"""Persistent ONNX Runtime session manager.

Provides warmed ORT CPU sessions, optional I/O binding, thread configuration,
and deterministic fallback logging on load or inference failure.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("tekmyra.onnx_runtime")


@dataclass(frozen=True)
class EngineLoad:
    """One attempt to bring up an ONNX session, and how it went."""

    model: str
    loaded: bool
    error: str | None = None


#: Every load attempt this process has made, in order. A run that degrades
#: silently is indistinguishable from a run that did not, which is the whole
#: reason this exists: the numbers do not depend on ONNX, but a RECORD that
#: cannot tell the two apart is not a record. Written only by ``_load``, which
#: every construction site routes through.
_LOAD_LEDGER: list[EngineLoad] = []


def record_engine_load(model: str, loaded: bool, error: str | None = None) -> None:
    """Append one load outcome to the ledger."""
    _LOAD_LEDGER.append(EngineLoad(model=model, loaded=loaded, error=error))


def engine_loads() -> list[EngineLoad]:
    """Every load attempt so far, oldest first."""
    return list(_LOAD_LEDGER)


def reset_engine_loads() -> None:
    """Clear the ledger. For tests and for callers measuring one scope."""
    _LOAD_LEDGER.clear()


def engine_availability() -> dict[str, Any]:
    """Serialisable summary of what was requested, what loaded, what did not.

    ``complete`` is the single question a reader needs answered: did every
    engine this run asked for actually come up? It is True when nothing was
    requested, because a run that never needed ONNX is not a degraded run.
    """
    loads = engine_loads()
    failed = [entry for entry in loads if not entry.loaded]
    # Counts are per ATTEMPT (a model reloaded per fixture is retried, and how
    # often is real information); the failure LIST is per distinct model, since
    # forty identical lines describe one broken engine, not forty.
    distinct: dict[str, dict[str, Any]] = {}
    for entry in failed:
        row = distinct.setdefault(
            entry.model, {"model": entry.model, "error": entry.error, "attempts": 0}
        )
        row["attempts"] += 1
    return {
        "requested": len(loads),
        "loaded": len(loads) - len(failed),
        "failed": len(failed),
        "complete": not failed,
        "distinct_models_failed": len(distinct),
        "failures": list(distinct.values()),
        # Which artifacts this run actually TOUCHED, by file name. Certification
        # cross-checks this against the bundle's real contents, so an artifact
        # that shipped but was never attempted cannot pass unremarked. Attempted
        # covers loaded AND failed: both are evidence the run reached for it.
        "attempted": sorted(
            {
                entry.model.split(".cache/", 1)[1]
                if ".cache/" in entry.model
                else Path(entry.model).name
                for entry in loads
            }
        ),
    }


@dataclass(frozen=True)
class OrtConfig:
    """ONNX Runtime session configuration.

    ``num_threads`` is a legacy alias used when the caller only cares about
    intra-op threads.  If ``intra_op_num_threads`` is 0, ``num_threads`` is
    used instead.  0 means "let ORT decide".
    """

    model_path: Path
    num_threads: int = 0
    inter_op_num_threads: int = 1
    intra_op_num_threads: int = 0
    graph_optimization_level: str = "ORT_ENABLE_ALL"
    use_io_binding: bool = True
    warmup_iterations: int = 3

    @classmethod
    def from_env(cls, model_path: Path) -> "OrtConfig":
        return cls(
            model_path=model_path,
            num_threads=int(os.environ.get("ORT_NUM_THREADS", "0")),
            inter_op_num_threads=int(os.environ.get("ORT_INTER_OP_THREADS", "1")),
            intra_op_num_threads=int(os.environ.get("ORT_INTRA_OP_THREADS", "0")),
            graph_optimization_level=os.environ.get("ORT_GRAPH_OPT", "ORT_ENABLE_ALL"),
            use_io_binding=os.environ.get("ORT_USE_IO_BINDING", "1") == "1",
            warmup_iterations=int(os.environ.get("ORT_WARMUP_ITER", "3")),
        )


# Announce an absent onnxruntime ONCE per process, not once per session
# construction — the manager is built per model per run, and a notice repeated
# 28 times is the traceback spam it replaced wearing a quieter costume.
_missing_onnxruntime_announced = False


class OnnxSessionManager:
    """Persistent ONNX Runtime session with CPU EP, optional I/O binding, and warmup."""

    def __init__(self, model_path: str | Path, config: OrtConfig | None = None) -> None:
        self._path = Path(model_path)
        self.config = config or OrtConfig.from_env(self._path)
        self._session: Any | None = None
        self._io_binding: Any | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._load()

    def _load(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            opts = ort.SessionOptions()
            opts.graph_optimization_level = getattr(
                ort.GraphOptimizationLevel, self.config.graph_optimization_level
            )
            intra = self.config.intra_op_num_threads or self.config.num_threads
            if intra > 0:
                opts.intra_op_num_threads = intra
            if self.config.inter_op_num_threads > 0:
                opts.inter_op_num_threads = self.config.inter_op_num_threads

            providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(
                str(self._path), sess_options=opts, providers=providers
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            if self.config.use_io_binding:
                self._io_binding = self._session.io_binding()
            self._warmup()
            logger.info("loaded onnx session", extra={"model": str(self._path)})
            record_engine_load(str(self._path), loaded=True)
        except ModuleNotFoundError as exc:
            # AN ABSENT onnxruntime IS THE DOCUMENTED CONFIGURATION, NOT AN
            # ERROR. The README's figures are measured on the joblib path with
            # no onnxruntime installed, and the extra is optional — yet this
            # branch used to log a full traceback PER FIXTURE: 28 stack traces
            # on the synthetic corpus's happy path, twice as much traceback as
            # report. One line, at INFO, once per process: the fallback is
            # working as designed, and the reader who wants the ONNX path is
            # told exactly how to get it.
            global _missing_onnxruntime_announced
            if not _missing_onnxruntime_announced:
                _missing_onnxruntime_announced = True
                logger.info(
                    "onnxruntime not installed; using the joblib/scikit-learn "
                    "path (the configuration the published figures were "
                    "measured on). Install the [transformer] extra for the "
                    "ONNX path.",
                    extra={"model": str(self._path)},
                )
            self._session = None
            record_engine_load(str(self._path), loaded=False, error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            # A PRESENT-BUT-BROKEN onnxruntime stays loud: this branch now
            # means an install that claims the capability and cannot deliver
            # it, which is a real environment defect worth a traceback.
            logger.error(
                "failed to load onnx session",
                extra={"error": str(exc), "model": str(self._path)},
                exc_info=True,
            )
            self._session = None
            record_engine_load(str(self._path), loaded=False, error=f"{type(exc).__name__}: {exc}")

    def _warmup(self) -> None:
        if self._session is None or self.config.warmup_iterations <= 0:
            return
        dtype_map = {
            "tensor(float)": np.float32,
            "tensor(double)": np.float64,
            "tensor(int8)": np.int8,
            "tensor(int16)": np.int16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(uint8)": np.uint8,
            "tensor(uint16)": np.uint16,
            "tensor(uint32)": np.uint32,
            "tensor(uint64)": np.uint64,
            "tensor(bool)": np.bool_,
        }
        input_dtypes = {
            inp.name: dtype_map.get(inp.type, np.float32)
            for inp in self._session.get_inputs()
        }
        for _ in range(self.config.warmup_iterations):
            feeds: dict[str, np.ndarray] = {}
            for name, inp in zip(self._input_names, self._session.get_inputs()):
                shape = [
                    1 if dim is None or not isinstance(dim, (int, np.integer)) else int(dim)
                    for dim in inp.shape
                ]
                feeds[name] = np.ones(shape, dtype=input_dtypes[name])
            self.run(feeds)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._session is None:
            raise RuntimeError("ONNX session not loaded")
        if self._io_binding is not None:
            self._io_binding.clear_binding_inputs()
            self._io_binding.clear_binding_outputs()
            for name, arr in inputs.items():
                self._io_binding.bind_input(
                    name,
                    device_type="cpu",
                    device_id=0,
                    element_type=arr.dtype,
                    shape=arr.shape,
                    buffer_ptr=arr.__array_interface__["data"][0],
                )
            for name in self._output_names:
                self._io_binding.bind_output(name)
            self._session.run_with_iobinding(self._io_binding)
            outputs = self._io_binding.copy_outputs_to_cpu()
            return {name: outputs[i] for i, name in enumerate(self._output_names)}
        outputs = self._session.run(self._output_names, inputs)
        return {name: out for name, out in zip(self._output_names, outputs)}
