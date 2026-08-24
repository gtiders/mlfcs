from __future__ import annotations

from mlfcs.fitting.backends.interface import FittingBasisBackend
from mlfcs.fitting.backends.result import BasisDiagnostics, BasisLoweringResult


def test_backend_result_objects_are_basis_independent():
    diagnostics = BasisDiagnostics(details={"backend": "test"})
    result = BasisLoweringResult(taylor_parameters=[1.0], diagnostics=diagnostics)

    assert result.diagnostics.details == {"backend": "test"}
    assert FittingBasisBackend is not None
