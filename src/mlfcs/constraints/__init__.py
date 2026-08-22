"""Physical IFC constraints independent of calculation backends."""

from mlfcs.constraints.rotational import (
    RotationalSumRuleDiagnostics,
    RotationalSumRuleResult,
    enforce_rotational_sum_rules,
)
from mlfcs.constraints.translational import (
    build_translational_constraints,
    maximum_acoustic_sum_rule_drift,
    maximum_constraint_residual,
    project_acoustic_sum_rule,
    project_parameters,
)

__all__ = [
    "RotationalSumRuleDiagnostics",
    "RotationalSumRuleResult",
    "build_translational_constraints",
    "enforce_rotational_sum_rules",
    "maximum_acoustic_sum_rule_drift",
    "maximum_constraint_residual",
    "project_acoustic_sum_rule",
    "project_parameters",
]
