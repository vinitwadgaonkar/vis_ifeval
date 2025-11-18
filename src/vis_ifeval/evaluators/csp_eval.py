"""CSP (Constraint Satisfaction Problem) evaluator for visual constraints."""

import logging
import math
import re
from typing import TYPE_CHECKING

from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.ocr_backend import TextBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CSPEvaluator(ConstraintEvaluator):
    """Evaluator for CSP (Constraint Satisfaction Problem) constraints."""

    def __init__(self, backend: TextBackend) -> None:
        """Initialize CSPEvaluator with an OCR backend.

        Args:
            backend: TextBackend instance for text extraction.
        """
        self.backend = backend

    def can_handle(self, constraint: dict) -> bool:
        """Check if constraint type is 'csp'."""
        return constraint.get("type") == "csp"

    def score(
        self, image: Image.Image, prompt: dict, constraint: dict
    ) -> float:
        """Score CSP constraint.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary with CSP specification.

        Returns:
            Score in [0, 1] indicating how well the CSP constraint is satisfied.
        """
        csp_kind = constraint.get("csp_kind", "")
        if not csp_kind:
            logger.warning("CSP constraint missing 'csp_kind' field")
            return 0.0

        try:
            # Extract text from image using OCR
            ocr_text = self.backend.extract_text(image).strip()
            if not ocr_text:
                logger.warning("OCR returned empty text")
                return 0.0

            # Parse numeric values from OCR text using field_map
            field_map = constraint.get("field_map", {})
            values = self._parse_values(ocr_text, field_map)

            # Route to appropriate CSP kind handler
            if csp_kind == "numeric_relation":
                return self._score_numeric_relation(values, constraint)
            elif csp_kind == "sum_equals":
                return self._score_sum_equals(values, constraint)
            elif csp_kind == "all_different":
                return self._score_all_different(values, constraint)
            elif csp_kind == "sorted":
                return self._score_sorted(values, constraint)
            else:
                logger.warning(f"Unknown csp_kind: {csp_kind}")
                return 0.0

        except Exception as e:
            logger.warning(f"CSP evaluation failed: {e}")
            return 0.0

    def _parse_values(self, ocr_text: str, field_map: dict[str, str]) -> dict[str, float]:
        """Parse numeric values from OCR text using field_map.

        Looks for patterns like "A: 3", "A = 3", "A 3", "row1: 10", etc.

        Args:
            ocr_text: Raw OCR text from the image.
            field_map: Dictionary mapping symbol names to field names to search for.

        Returns:
            Dictionary mapping symbol names to parsed float values.
        """
        values: dict[str, float] = {}
        text_lower = ocr_text.lower()

        for symbol, field_name in field_map.items():
            # Try multiple patterns: "A: 3", "A = 3", "A 3", "A:3", etc.
            # Escape field name for regex
            field_escaped = re.escape(field_name.lower())
            patterns = [
                rf"{field_escaped}\s*:\s*([0-9.]+)",  # "A: 3"
                rf"{field_escaped}\s*=\s*([0-9.]+)",  # "A = 3"
                rf"{field_escaped}\s+([0-9.]+)",      # "A 3"
                rf"{field_escaped}:([0-9.]+)",        # "A:3"
            ]

            found = False
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        value = float(match.group(1))
                        values[symbol] = value
                        found = True
                        break
                    except ValueError:
                        continue

            if not found:
                logger.debug(f"Could not parse value for symbol '{symbol}' (field '{field_name}')")

        return values

    def _score_numeric_relation(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score numeric relation constraint (e.g., A < B).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'lhs', 'rhs', 'relation'.

        Returns:
            Score in [0, 1]. 1.0 if relation is satisfied, 0.0 otherwise.
        """
        lhs = constraint.get("lhs")
        rhs = constraint.get("rhs")
        relation = constraint.get("relation")

        if not lhs or not rhs or not relation:
            logger.warning("numeric_relation constraint missing 'lhs', 'rhs', or 'relation'")
            return 0.0

        if lhs not in values or rhs not in values:
            logger.warning(f"Missing values for numeric_relation: lhs={lhs}, rhs={rhs}")
            return 0.0

        val_lhs = values[lhs]
        val_rhs = values[rhs]

        # Evaluate relation
        if relation == "<":
            satisfied = val_lhs < val_rhs
        elif relation == "<=":
            satisfied = val_lhs <= val_rhs
        elif relation == "==":
            # Use small epsilon for equality
            satisfied = abs(val_lhs - val_rhs) < 1e-6
        elif relation == ">=":
            satisfied = val_lhs >= val_rhs
        elif relation == ">":
            satisfied = val_lhs > val_rhs
        elif relation == "!=":
            satisfied = abs(val_lhs - val_rhs) >= 1e-6
        else:
            logger.warning(f"Unknown relation operator: {relation}")
            return 0.0

        return 1.0 if satisfied else 0.0

    def _score_sum_equals(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score sum_equals constraint (e.g., A + B + C = T).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables' and 'target'.

        Returns:
            Score in [0, 1] based on relative error.
        """
        variables = constraint.get("variables", [])
        target_symbol = constraint.get("target")

        if not variables or not target_symbol:
            logger.warning("sum_equals constraint missing 'variables' or 'target'")
            return 0.0

        # Check all variables are present
        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for sum_equals variables: {missing}")
            return 0.0

        if target_symbol not in values:
            logger.warning(f"Missing target value for sum_equals: {target_symbol}")
            return 0.0

        # Compute sum
        sum_value = sum(values[v] for v in variables)
        target_value = values[target_symbol]

        # Compute relative error
        rel_err = abs(sum_value - target_value) / max(1.0, abs(target_value))

        # Convert to score using exponential decay
        alpha = 3.0
        score = math.exp(-alpha * rel_err)
        return max(0.0, min(1.0, score))

    def _score_all_different(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score all_different constraint (all values must be distinct).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables'.

        Returns:
            Score in [0, 1]. 1.0 if all values are distinct, 0.0 otherwise.
        """
        variables = constraint.get("variables", [])

        if not variables:
            logger.warning("all_different constraint missing 'variables'")
            return 0.0

        # Check all variables are present
        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for all_different variables: {missing}")
            return 0.0

        # Check if all values are distinct
        value_list = [values[v] for v in variables]
        unique_values = set(value_list)
        satisfied = len(unique_values) == len(variables)

        return 1.0 if satisfied else 0.0

    def _score_sorted(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score sorted constraint (values must be in order).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables', 'order', 'strict'.

        Returns:
            Score in [0, 1]. 1.0 if values are in correct order, 0.0 otherwise.
        """
        variables = constraint.get("variables", [])
        order = constraint.get("order", "ascending")
        strict = constraint.get("strict", True)

        if not variables:
            logger.warning("sorted constraint missing 'variables'")
            return 0.0

        # Check all variables are present
        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for sorted variables: {missing}")
            return 0.0

        # Build list of values in order
        value_list = [values[v] for v in variables]

        # Check ordering
        if order == "ascending":
            if strict:
                # Strict: L[0] < L[1] < ... < L[n-1]
                satisfied = all(value_list[i] < value_list[i + 1] for i in range(len(value_list) - 1))
            else:
                # Non-strict: L[0] <= L[1] <= ...
                satisfied = all(value_list[i] <= value_list[i + 1] for i in range(len(value_list) - 1))
        elif order == "descending":
            if strict:
                # Strict: L[0] > L[1] > ... > L[n-1]
                satisfied = all(value_list[i] > value_list[i + 1] for i in range(len(value_list) - 1))
            else:
                # Non-strict: L[0] >= L[1] >= ...
                satisfied = all(value_list[i] >= value_list[i + 1] for i in range(len(value_list) - 1))
        else:
            logger.warning(f"Unknown order: {order}")
            return 0.0

        return 1.0 if satisfied else 0.0

