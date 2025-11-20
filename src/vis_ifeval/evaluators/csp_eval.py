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
            elif csp_kind == "range":
                return self._score_range(values, constraint)
            elif csp_kind == "equals":
                return self._score_equals(values, constraint)
            elif csp_kind == "product_equals":
                return self._score_product_equals(values, constraint)
            elif csp_kind == "difference_equals":
                return self._score_difference_equals(values, constraint)
            elif csp_kind == "ratio":
                return self._score_ratio(values, constraint)
            elif csp_kind == "min_value":
                return self._score_min_value(values, constraint)
            elif csp_kind == "max_value":
                return self._score_max_value(values, constraint)
            elif csp_kind == "chain_relation":
                return self._score_chain_relation(values, constraint)
            elif csp_kind == "modulo":
                return self._score_modulo(values, constraint)
            else:
                logger.warning(f"Unknown csp_kind: {csp_kind}")
                return 0.0

        except Exception as e:
            logger.warning(f"CSP evaluation failed: {e}")
            return 0.0

    def _parse_values(self, ocr_text: str, field_map: dict[str, str]) -> dict[str, float]:
        """Parse numeric values from OCR text using field_map.

        Looks for patterns like "A: 3", "A = 3", "A 3", "row1: 10", etc.
        Supports negative numbers, decimals, and units (strips units).

        Args:
            ocr_text: Raw OCR text from the image.
            field_map: Dictionary mapping symbol names to field names to search for.

        Returns:
            Dictionary mapping symbol names to parsed float values.
        """
        values: dict[str, float] = {}
        # Remove markdown formatting and normalize whitespace
        text_clean = re.sub(r'\*\*|\*|#+', '', ocr_text)  # Remove markdown
        text_lower = text_clean.lower()

        for symbol, field_name in field_map.items():
            # Try multiple patterns: "A: 3", "A = 3", "A 3", "A:3", "A: -3.5", etc.
            # Escape field name for regex
            field_escaped = re.escape(field_name.lower())
            
            # Enhanced patterns supporting negative numbers, decimals, and units
            # Also handle multi-line patterns (label on one line, number on next)
            patterns = [
                # Standard patterns with optional negative sign and decimals
                rf"{field_escaped}\s*:\s*(-?\d+\.?\d*)",  # "A: 3", "A: -3.5"
                rf"{field_escaped}\s*=\s*(-?\d+\.?\d*)",  # "A = 3", "A = -3.5"
                rf"{field_escaped}\s+(-?\d+\.?\d*)",      # "A 3", "A -3.5"
                rf"{field_escaped}:(-?\d+\.?\d*)",        # "A:3", "A:-3.5"
                # Patterns with units (kg, g, ml, etc.) - extract number before unit
                rf"{field_escaped}\s*:\s*(-?\d+\.?\d*)\s*(?:kg|g|ml|l|mg|oz|lb|cm|m|in|ft)",  # "A: 3 kg"
                rf"{field_escaped}\s*=\s*(-?\d+\.?\d*)\s*(?:kg|g|ml|l|mg|oz|lb|cm|m|in|ft)",
                # Table-like patterns
                rf"{field_escaped}\s*\|\s*(-?\d+\.?\d*)",  # "A | 3" (table format)
                rf"{field_escaped}\s*\t\s*(-?\d+\.?\d*)",  # "A \t 3" (tab-separated)
            ]

            found = False
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.MULTILINE | re.DOTALL)
                if match:
                    try:
                        value_str = match.group(1)
                        value = float(value_str)
                        values[symbol] = value
                        found = True
                        break
                    except ValueError:
                        continue
            
            # If not found, try multi-line patterns (label on one line, number on next line)
            if not found:
                multiline_patterns = [
                    rf"{field_escaped}\s*:?\s*\n\s*(-?\d+\.?\d*)",  # "A:\n15" or "A\n15"
                    rf"{field_escaped}\s*=\s*\n\s*(-?\d+\.?\d*)",   # "Total =\n78"
                ]
                for pattern in multiline_patterns:
                    match = re.search(pattern, text_lower, re.MULTILINE | re.DOTALL)
                    if match:
                        try:
                            value_str = match.group(1)
                            value = float(value_str)
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

    def _score_range(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score range constraint (value must be within [min, max]).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variable', 'min', 'max'.

        Returns:
            Score in [0, 1]. 1.0 if value is in range, 0.0 otherwise.
        """
        variable = constraint.get("variable")
        min_val = constraint.get("min")
        max_val = constraint.get("max")

        if not variable or min_val is None or max_val is None:
            logger.warning("range constraint missing 'variable', 'min', or 'max'")
            return 0.0

        if variable not in values:
            logger.warning(f"Missing value for range variable: {variable}")
            return 0.0

        value = values[variable]
        satisfied = min_val <= value <= max_val

        return 1.0 if satisfied else 0.0

    def _score_equals(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score equals constraint (all variables must have the same value).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables'.

        Returns:
            Score in [0, 1]. 1.0 if all values are equal, 0.0 otherwise.
        """
        variables = constraint.get("variables", [])

        if not variables:
            logger.warning("equals constraint missing 'variables'")
            return 0.0

        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for equals variables: {missing}")
            return 0.0

        value_list = [values[v] for v in variables]
        if len(value_list) < 2:
            return 1.0

        # Check if all values are equal (within epsilon)
        first_val = value_list[0]
        epsilon = constraint.get("tolerance", 1e-6)
        satisfied = all(abs(v - first_val) < epsilon for v in value_list[1:])

        return 1.0 if satisfied else 0.0

    def _score_product_equals(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score product_equals constraint (e.g., A * B = C).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables' and 'target'.

        Returns:
            Score in [0, 1] based on relative error.
        """
        variables = constraint.get("variables", [])
        target_symbol = constraint.get("target")

        if not variables or not target_symbol:
            logger.warning("product_equals constraint missing 'variables' or 'target'")
            return 0.0

        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for product_equals variables: {missing}")
            return 0.0

        if target_symbol not in values:
            logger.warning(f"Missing target value for product_equals: {target_symbol}")
            return 0.0

        # Compute product
        product = 1.0
        for v in variables:
            product *= values[v]

        target_value = values[target_symbol]

        # Compute relative error
        rel_err = abs(product - target_value) / max(1.0, abs(target_value))

        # Convert to score using exponential decay
        alpha = 3.0
        score = math.exp(-alpha * rel_err)
        return max(0.0, min(1.0, score))

    def _score_difference_equals(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score difference_equals constraint (e.g., A - B = C).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'lhs', 'rhs', 'target'.

        Returns:
            Score in [0, 1] based on relative error.
        """
        lhs = constraint.get("lhs")
        rhs = constraint.get("rhs")
        target_symbol = constraint.get("target")

        if not lhs or not rhs or not target_symbol:
            logger.warning("difference_equals constraint missing 'lhs', 'rhs', or 'target'")
            return 0.0

        if lhs not in values or rhs not in values or target_symbol not in values:
            logger.warning(f"Missing values for difference_equals: lhs={lhs}, rhs={rhs}, target={target_symbol}")
            return 0.0

        difference = values[lhs] - values[rhs]
        target_value = values[target_symbol]

        # Compute relative error
        rel_err = abs(difference - target_value) / max(1.0, abs(target_value))

        # Convert to score using exponential decay
        alpha = 3.0
        score = math.exp(-alpha * rel_err)
        return max(0.0, min(1.0, score))

    def _score_ratio(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score ratio constraint (e.g., A / B = ratio).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'numerator', 'denominator', 'target'.

        Returns:
            Score in [0, 1] based on relative error.
        """
        numerator = constraint.get("numerator")
        denominator = constraint.get("denominator")
        target_symbol = constraint.get("target")

        if not numerator or not denominator or not target_symbol:
            logger.warning("ratio constraint missing 'numerator', 'denominator', or 'target'")
            return 0.0

        if numerator not in values or denominator not in values or target_symbol not in values:
            logger.warning(f"Missing values for ratio: num={numerator}, den={denominator}, target={target_symbol}")
            return 0.0

        if abs(values[denominator]) < 1e-10:
            logger.warning("Division by zero in ratio constraint")
            return 0.0

        ratio = values[numerator] / values[denominator]
        target_value = values[target_symbol]

        # Compute relative error
        rel_err = abs(ratio - target_value) / max(1.0, abs(target_value))

        # Convert to score using exponential decay
        alpha = 3.0
        score = math.exp(-alpha * rel_err)
        return max(0.0, min(1.0, score))

    def _score_min_value(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score min_value constraint (variable must be >= minimum).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variable' and 'min'.

        Returns:
            Score in [0, 1]. 1.0 if value >= min, 0.0 otherwise.
        """
        variable = constraint.get("variable")
        min_val = constraint.get("min")

        if not variable or min_val is None:
            logger.warning("min_value constraint missing 'variable' or 'min'")
            return 0.0

        if variable not in values:
            logger.warning(f"Missing value for min_value variable: {variable}")
            return 0.0

        value = values[variable]
        satisfied = value >= min_val

        return 1.0 if satisfied else 0.0

    def _score_max_value(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score max_value constraint (variable must be <= maximum).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variable' and 'max'.

        Returns:
            Score in [0, 1]. 1.0 if value <= max, 0.0 otherwise.
        """
        variable = constraint.get("variable")
        max_val = constraint.get("max")

        if not variable or max_val is None:
            logger.warning("max_value constraint missing 'variable' or 'max'")
            return 0.0

        if variable not in values:
            logger.warning(f"Missing value for max_value variable: {variable}")
            return 0.0

        value = values[variable]
        satisfied = value <= max_val

        return 1.0 if satisfied else 0.0

    def _score_chain_relation(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score chain_relation constraint (e.g., A < B < C < D).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'variables' and 'relation'.

        Returns:
            Score in [0, 1]. 1.0 if chain is satisfied, 0.0 otherwise.
        """
        variables = constraint.get("variables", [])
        relation = constraint.get("relation", "<")
        strict = constraint.get("strict", True)

        if not variables or len(variables) < 2:
            logger.warning("chain_relation constraint missing 'variables' or has < 2 variables")
            return 0.0

        missing = [v for v in variables if v not in values]
        if missing:
            logger.warning(f"Missing values for chain_relation variables: {missing}")
            return 0.0

        value_list = [values[v] for v in variables]

        # Check chain relation
        if relation == "<":
            if strict:
                satisfied = all(value_list[i] < value_list[i + 1] for i in range(len(value_list) - 1))
            else:
                satisfied = all(value_list[i] <= value_list[i + 1] for i in range(len(value_list) - 1))
        elif relation == ">":
            if strict:
                satisfied = all(value_list[i] > value_list[i + 1] for i in range(len(value_list) - 1))
            else:
                satisfied = all(value_list[i] >= value_list[i + 1] for i in range(len(value_list) - 1))
        elif relation == "==":
            epsilon = constraint.get("tolerance", 1e-6)
            satisfied = all(abs(value_list[i] - value_list[i + 1]) < epsilon for i in range(len(value_list) - 1))
        else:
            logger.warning(f"Unknown chain relation: {relation}")
            return 0.0

        return 1.0 if satisfied else 0.0

    def _score_modulo(
        self, values: dict[str, float], constraint: dict
    ) -> float:
        """Score modulo constraint (e.g., A % B = C).

        Args:
            values: Dictionary mapping symbols to numeric values.
            constraint: Constraint dictionary with 'dividend', 'divisor', 'target'.

        Returns:
            Score in [0, 1]. 1.0 if modulo is satisfied, 0.0 otherwise.
        """
        dividend = constraint.get("dividend")
        divisor = constraint.get("divisor")
        target_symbol = constraint.get("target")

        if not dividend or not divisor or not target_symbol:
            logger.warning("modulo constraint missing 'dividend', 'divisor', or 'target'")
            return 0.0

        if dividend not in values or divisor not in values or target_symbol not in values:
            logger.warning(f"Missing values for modulo: dividend={dividend}, divisor={divisor}, target={target_symbol}")
            return 0.0

        if abs(values[divisor]) < 1e-10:
            logger.warning("Division by zero in modulo constraint")
            return 0.0

        # Compute modulo (handles negative numbers correctly)
        remainder = values[dividend] % values[divisor]
        target_value = values[target_symbol]

        # Check equality (modulo can have slight floating point issues)
        epsilon = constraint.get("tolerance", 1e-6)
        satisfied = abs(remainder - target_value) < epsilon

        return 1.0 if satisfied else 0.0

