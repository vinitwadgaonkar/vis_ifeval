"""Negative constraint evaluator using CLIP."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.clip_utils import ClipModelWrapper

logger = logging.getLogger(__name__)


class NegativeEvaluator(ConstraintEvaluator):
    """
    Evaluates negative (forbidden) concepts using CLIP.

    For now, supports concept == "sugar_drink".
    """

    def __init__(self, clip: ClipModelWrapper) -> None:
        """Initialize NegativeEvaluator with CLIP wrapper.

        Args:
            clip: ClipModelWrapper instance for image-text similarity.
        """
        self.clip = clip
        self._warned_disabled = False

        # Predefine prompts for the sugar_drink concept.
        self.concept_prompts = {
            "sugar_drink": [
                "a bottle of sugary soda",
                "a sweet soft drink",
                "a soda bottle with sugar",
                "a sugary beverage",
            ]
        }

    def can_handle(self, constraint: Dict[str, Any]) -> bool:
        """Check if constraint type is 'negative'."""
        return constraint.get("type") == "negative"

    def score(
        self, image: "Image.Image", prompt: Dict[str, Any], constraint: Dict[str, Any]
    ) -> float:
        """Score negative constraint using CLIP.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary with 'concept' field.

        Returns:
            Score in [0, 1] where 1.0 means the forbidden concept is absent.
        """
        concept = constraint.get("concept")
        if concept not in self.concept_prompts:
            # Unknown concept; for now treat as not violated.
            logger.warning("NegativeEvaluator: unknown concept %r, returning 1.0", concept)
            return 1.0

        if not self.clip.enabled:
            if not self._warned_disabled:
                logger.warning(
                    "NegativeEvaluator: CLIP is disabled, returning 1.0 for all negative constraints"
                )
                self._warned_disabled = True
            return 1.0

        texts = self.concept_prompts[concept]
        sims = self.clip.image_text_similarities(image, texts)
        if sims is None or len(sims) == 0:
            return 1.0

        max_sim = max(sims)  # cosine in [-1, 1]

        # Map similarity to [0,1] where high similarity => low score.
        # If max_sim <= 0, treat as safe (score ~1).
        # If max_sim >= 1, score ~0.
        sim_pos = max(0.0, max_sim)  # [0,1]
        score = 1.0 - sim_pos
        score = float(max(0.0, min(1.0, score)))

        return score
