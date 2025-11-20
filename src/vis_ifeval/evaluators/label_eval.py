"""Nutrition label evaluator."""

import logging
import math
import re
from typing import TYPE_CHECKING

from Levenshtein import distance as levenshtein_distance
from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.ocr_backend import TextBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LabelEvaluator(ConstraintEvaluator):
    """Evaluator for nutrition label constraints."""

    def __init__(self, backend: TextBackend) -> None:
        """Initialize LabelEvaluator with an OCR backend.

        Args:
            backend: TextBackend instance for text extraction.
        """
        self.backend = backend

    def can_handle(self, constraint: dict) -> bool:
        """Check if constraint type is 'table_slot'."""
        return constraint.get("type") == "table_slot"

    def _crop_label_region(self, image: Image.Image) -> Image.Image:
        """Crop an approximate label region from the image.

        Args:
            image: Full image.

        Returns:
            Cropped image focusing on the label region.
        """
        w, h = image.size
        # Focus on center band; tweak later if needed
        x0, y0 = int(0.15 * w), int(0.2 * h)
        x1, y1 = int(0.85 * w), int(0.9 * h)
        return image.crop((x0, y0, x1, y1))

    def _parse_fields(self, raw_text: str) -> dict[str, str]:
        """Parse nutrition label fields from raw OCR text.

        Args:
            raw_text: Raw OCR text from the label.

        Returns:
            Dictionary mapping field keys to normalized values.
        """
        text_lower = raw_text.lower()
        lines = [line.strip() for line in text_lower.split("\n") if line.strip()]
        parsed: dict[str, str] = {}

        for line in lines:
            # Serving size - handle OCR errors like "servina size", "serving siz", etc.
            # Use fuzzy matching for "serving size"
            if "serv" in line and ("size" in line or "siz" in line):
                # Extract value - look for pattern like "serving size: 250 ml" or "servina size 250 ml"
                # More flexible pattern that handles OCR errors
                match = re.search(r"serv[^:]*siz[^:]*[:\s]+([0-9.]+)\s*(ml|g|oz|l)", line, re.IGNORECASE)
                if match:
                    parsed["serving_size"] = f"{match.group(1)} {match.group(2).lower()}"

            # Calories
            if line.startswith("calories") or "calories" in line:
                match = re.search(r"calories[:\s]+([0-9]+)", line)
                if match:
                    parsed["calories"] = match.group(1)

            # Total fat
            if "total fat" in line:
                match = re.search(r"total fat[:\s]+([0-9.]+)\s*(g|mg)", line)
                if match:
                    parsed["total_fat"] = f"{match.group(1)} {match.group(2)}"

            # Sodium
            if "sodium" in line.lower():
                # Extract mg value - handle various formats: "Sodium: 200 mg", "Sodium 200 mg", "Sodium-200 mg", table format "| Sodium | 50mg |"
                # Look for number followed by "mg" anywhere in the line after "sodium"
                mg_match = re.search(r"sodium[:\s\-|]*.*?([0-9.]+)\s*mg", line, re.IGNORECASE)
                if mg_match:
                    parsed["sodium_mg"] = f"{mg_match.group(1)} mg"
                # Extract %DV - handle OCR errors like "2g" instead of "2%"
                # First try to find a percentage sign
                dv_match = re.search(r"([0-9.]+)\s*%", line)
                if dv_match:
                    parsed["sodium_dv_percent"] = f"{dv_match.group(1)}%"
                elif not parsed.get("sodium_dv_percent"):
                    # If no % found, look for a small number (0-20) that might be a percentage
                    # This handles OCR errors where "%" is misread as "g" or other characters
                    small_num_match = re.search(r"([0-9]{1,2})\s*[g%]", line)
                    if small_num_match and int(small_num_match.group(1)) <= 20:
                        parsed["sodium_dv_percent"] = f"{small_num_match.group(1)}%"

            # Total carbohydrate
            if "total carbohydrate" in line or "total carb" in line:
                match = re.search(
                    r"total (?:carbohydrate|carb)[:\s]+([0-9.]+)\s*(g|mg)", line
                )
                if match:
                    parsed["total_carbohydrate"] = f"{match.group(1)} {match.group(2)}"

            # Protein
            if line.startswith("protein") or "protein" in line:
                match = re.search(r"protein[:\s]+([0-9.]+)\s*(g|mg)", line)
                if match:
                    parsed["protein"] = f"{match.group(1)} {match.group(2)}"

        return parsed

    def score(
        self, image: Image.Image, prompt: dict, constraint: dict
    ) -> float:
        """Score nutrition label constraint.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary with 'field' and 'target'.

        Returns:
            Score in [0, 1] indicating how well the field matches the target.
        """
        field = constraint.get("field", "").lower()
        target = constraint.get("target", "")

        if not field or not target:
            logger.warning("Label constraint missing 'field' or 'target'")
            return 0.0

        try:
            # Crop label region and extract text
            label_image = self._crop_label_region(image)
            raw_text = self.backend.extract_text(label_image)
            parsed = self._parse_fields(raw_text)

            # Map constraint field names to parsed keys
            field_map = {
                "serving size": "serving_size",
                "calories": "calories",
                "total fat": "total_fat",
                "sodium": "sodium_mg",
                "sodium %dv": "sodium_dv_percent",
                "total carbohydrate": "total_carbohydrate",
                "protein": "protein",
            }

            parsed_key = field_map.get(field, field.lower().replace(" ", "_"))
            if parsed_key not in parsed:
                logger.debug(f"Field '{field}' not found in parsed label")
                return 0.0

            parsed_value = parsed[parsed_key]
            target_normalized = target.lower().strip()
            parsed_normalized = parsed_value.lower().strip()

            # Compute text-based score using CER
            edit_dist = levenshtein_distance(parsed_normalized, target_normalized)
            max_len = max(len(target_normalized), 1)
            cer = edit_dist / max_len
            s_text = math.exp(-3.0 * cer)

            # Try numeric refinement for numeric fields
            numeric_fields = {"calories", "sodium_mg", "sodium_dv_percent"}
            if parsed_key in numeric_fields:
                try:
                    # Extract numbers
                    parsed_num = float(re.search(r"([0-9.]+)", parsed_normalized).group(1))
                    target_num = float(re.search(r"([0-9.]+)", target_normalized).group(1))

                    rel_err = abs(parsed_num - target_num) / max(1.0, abs(target_num))
                    s_num = math.exp(-4.0 * rel_err)
                    score = (s_text ** 0.6) * (s_num ** 0.4)
                except (ValueError, AttributeError):
                    score = s_text
            else:
                score = s_text

            return max(0.0, min(1.0, float(score)))

        except Exception as e:
            logger.warning(f"Label evaluation failed: {e}")
            return 0.0
