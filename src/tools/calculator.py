"""Safe calculator tool using AST-based expression evaluation."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Sequence

import structlog

logger = structlog.get_logger(__name__)

# Supported binary operators
_BINARY_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
_UNARY_OPS: dict[type, object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Safe math constants and single-arg functions
_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_SAFE_FUNCS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

# Maximum exponent to prevent resource exhaustion
_MAX_POWER = 1000


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node into a float value.

    Only allows arithmetic expressions — no attribute access, subscripts,
    assignments, or arbitrary function calls.
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]

    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return float(op_fn(_safe_eval_node(node.operand)))  # type: ignore[operator]

    if isinstance(node, ast.BinOp):
        op_fn = _BINARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if isinstance(node.op, ast.Pow) and right > _MAX_POWER:
            raise ValueError(f"Exponent {right} exceeds maximum allowed ({_MAX_POWER})")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError("Division by zero")
        return float(op_fn(left, right))  # type: ignore[operator]

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named function calls are allowed")
        func = _SAFE_FUNCS.get(node.func.id)
        if func is None:
            raise ValueError(f"Unsupported function: {node.func.id}")
        args = [_safe_eval_node(a) for a in node.args]
        return float(func(*args))  # type: ignore[operator]

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


class CalculatorTool:
    """Safe mathematical calculator — no exec/eval, uses AST parsing."""

    @staticmethod
    def calculate(expression: str) -> float:
        """Safely evaluate a mathematical expression.

        Supports: +, -, *, /, //, %, **, parentheses, and safe math
        functions (sqrt, log, abs, round, sin, cos, tan, ceil, floor).

        Args:
            expression: A mathematical expression string.

        Returns:
            The computed float result.

        Raises:
            ValueError: If the expression contains disallowed constructs.
            ZeroDivisionError: On division by zero.
        """
        if not expression or not expression.strip():
            raise ValueError("Empty expression")

        logger.debug("calculator.calculate", expression=expression)
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid expression syntax: {exc}") from exc

        result = _safe_eval_node(tree)
        logger.debug("calculator.result", result=result)
        return result

    @staticmethod
    def sum_values(values: Sequence[float]) -> float:
        """Sum a sequence of numeric values.

        Args:
            values: Iterable of numbers.

        Returns:
            The sum as a float.
        """
        total = math.fsum(float(v) for v in values)
        logger.debug("calculator.sum", count=len(values), total=total)
        return total

    @staticmethod
    def validate_total(
        items: Sequence[float],
        expected_total: float,
        tolerance: float = 0.01,
    ) -> bool:
        """Check whether summed items match an expected total within tolerance.

        Args:
            items: Individual amounts to sum.
            expected_total: The expected total.
            tolerance: Acceptable absolute difference (default 0.01).

        Returns:
            True if the difference is within tolerance.
        """
        actual = math.fsum(float(v) for v in items)
        diff = abs(actual - expected_total)
        ok = diff <= tolerance
        logger.debug(
            "calculator.validate_total",
            actual=actual,
            expected=expected_total,
            diff=diff,
            passed=ok,
        )
        return ok

    @staticmethod
    def percentage_change(old: float, new: float) -> float:
        """Calculate percentage change from old to new value.

        Args:
            old: The original value.
            new: The new value.

        Returns:
            Percentage change as a float (e.g. 50.0 for a 50% increase).

        Raises:
            ZeroDivisionError: If old is zero.
        """
        if old == 0:
            raise ZeroDivisionError("Cannot compute percentage change from zero")
        change = ((new - old) / abs(old)) * 100.0
        logger.debug("calculator.percentage_change", old=old, new=new, change=change)
        return change
