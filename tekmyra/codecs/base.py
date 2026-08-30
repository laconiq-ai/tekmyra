"""Base types for deterministic structure codecs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CodecResult:
    """Result of a deterministic compression codec.

    Codecs are fail-closed: on any structural error or preservation failure
    they set ``success=False`` so the runtime can fall back to a safer route.
    """

    success: bool
    output: str
    ratio: float
    error: str | None = None
    preserved_spans: int = 0
    dropped_spans: int = 0
    learned_compressor_invoked: bool = False
    # THE BUDGET ASKED FOR AND THE BUDGET HONOURED (Vega ruling 47).
    #
    # A fixture whose locked spans saturate its budget cannot reach that budget
    # while honouring the entity retention contract, so the producer backs off
    # to the tightest budget it CAN honour. Reporting the requested figure alone
    # would state a target that was not met as though it were; carrying both
    # makes the degradation legible at the row it happened on. None means the
    # codec has no budget notion, which is most of them.
    budget_requested: float | None = None
    budget_achieved: float | None = None
    # Producer-side selection detail, carried so the benchmark row can show
    # whether a guarantee is binding PREVENTIVELY. With the entity floor
    # honoured during selection, the verifier's check stops rejecting anything —
    # and a check that never fails looks exactly like a check that is not there.
    # These fields are the difference. A mapping rather than more scalars
    # because the contents are one codec's business; the reporting layer names
    # the keys it expects.
    selection_diagnostics: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.ratio < 0.0:
            raise ValueError("compression ratio cannot be negative")
