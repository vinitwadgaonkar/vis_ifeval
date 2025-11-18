"""Composition evaluator using CLIP heuristics."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.clip_utils import ClipModelWrapper

logger = logging.getLogger(__name__)


class CompositionEvaluator(ConstraintEvaluator):
    """
    CLIP-based heuristic evaluator for composition constraints.

    - count: use "one/two/three/four ..." prompts and compare CLIP sims.
    - attribute: check similarity to "a {attribute} {object}" vs "{object}".
    - state: check similarity to "a {state} {object}" vs "{object}".
    - spatial: currently a stub (returns 0.0 with warning).
    """

    def __init__(self, clip: ClipModelWrapper) -> None:
        """Initialize CompositionEvaluator with CLIP wrapper.

        Args:
            clip: ClipModelWrapper instance for image-text similarity.
        """
        self.clip = clip
        self._warned_disabled = False

    def can_handle(self, constraint: Dict[str, Any]) -> bool:
        """Check if constraint type is composition-related."""
        return constraint.get("type") in {"count", "attribute", "spatial", "state"}

    def _check_clip_enabled(self) -> bool:
        """Check if CLIP is enabled, log warning if not."""
        if not self.clip.enabled:
            if not self._warned_disabled:
                logger.warning(
                    "CompositionEvaluator: CLIP is disabled, returning 0.0 for composition constraints"
                )
                self._warned_disabled = True
            return False
        return True

    # ---------- COUNT ----------

    def _score_count(self, image: "Image.Image", constraint: Dict[str, Any]) -> float:
        """
        Approximate count by comparing CLIP sims for different count captions.

        Assumes constraint has:
        - object: e.g. "blue mug"
        - target: integer count (1-4 or 1-5).
        """
        if not self._check_clip_enabled():
            return 0.0

        obj = constraint.get("object", "object")
        target = int(constraint.get("target", 1))

        # Limit to a reasonable range.
        counts = sorted({max(1, target - 1), target, target + 1, 1, 2, 3, 4})
        counts = [c for c in counts if 1 <= c <= 5]

        texts = [f"{c} {obj}" if c > 1 else f"one {obj}" for c in counts]
        sims = self.clip.image_text_similarities(image, texts)
        if sims is None or len(sims) == 0:
            return 0.0

        # Identify target index.
        try:
            target_idx = counts.index(target)
        except ValueError:
            # Should not happen, but be safe.
            return 0.0

        sim_target = sims[target_idx]
        sim_others = [s for i, s in enumerate(sims) if i != target_idx]
        best_other = max(sim_others) if sim_others else -1.0

        margin = sim_target - best_other  # positive if target wins.
        # Map margin to [0,1] via logistic.
        score = 1.0 / (1.0 + math.exp(-8.0 * margin))
        return float(max(0.0, min(1.0, score)))

    # ---------- ATTRIBUTE ----------

    def _score_attribute(self, image: "Image.Image", constraint: Dict[str, Any]) -> float:
        """
        Check if object appears with given attribute.

        Assumes:
        - constraint["object"], e.g. "mug"
        - constraint["attribute"], e.g. "blue"
        """
        if not self._check_clip_enabled():
            return 0.0

        obj = constraint.get("object", "object")
        attr = constraint.get("attribute", "")
        if not attr:
            return 0.0

        text_attr = f"a {attr} {obj}"
        text_plain = f"a {obj}"

        sims = self.clip.image_text_similarities(image, [text_attr, text_plain])
        if sims is None or len(sims) != 2:
            return 0.0

        sim_attr, sim_plain = sims
        margin = sim_attr - sim_plain
        score = 1.0 / (1.0 + math.exp(-8.0 * margin))
        return float(max(0.0, min(1.0, score)))

    # ---------- STATE ----------

    def _score_state(self, image: "Image.Image", constraint: Dict[str, Any]) -> float:
        """
        Simple state check like "empty cup", "broken glass", etc.

        Assumes:
        - constraint["object"]
        - constraint["state"]
        """
        if not self._check_clip_enabled():
            return 0.0

        obj = constraint.get("object", "object")
        state = constraint.get("state", "")
        if not state:
            return 0.0

        text_state = f"a {state} {obj}"
        text_plain = f"a {obj}"

        sims = self.clip.image_text_similarities(image, [text_state, text_plain])
        if sims is None or len(sims) != 2:
            return 0.0

        sim_state, sim_plain = sims
        margin = sim_state - sim_plain
        score = 1.0 / (1.0 + math.exp(-8.0 * margin))
        return float(max(0.0, min(1.0, score)))

    # ---------- SPATIAL (TODO) ----------

    def _score_spatial(self, image: "Image.Image", constraint: Dict[str, Any]) -> float:
        """
        Placeholder for future spatial relation evaluation (needs detection / grounding).
        """
        logger.warning("CompositionEvaluator: spatial constraints not yet implemented, returning 0.0")
        return 0.0

    def score(
        self, image: "Image.Image", prompt: Dict[str, Any], constraint: Dict[str, Any]
    ) -> float:
        """Score composition constraint.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary.

        Returns:
            Score in [0, 1] indicating how well the constraint is satisfied.
        """
        ctype = constraint.get("type")
        if ctype == "count":
            return self._score_count(image, constraint)
        if ctype == "attribute":
            return self._score_attribute(image, constraint)
        if ctype == "state":
            return self._score_state(image, constraint)
        if ctype == "spatial":
            return self._score_spatial(image, constraint)
        return 0.0
