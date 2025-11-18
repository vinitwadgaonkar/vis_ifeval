"""Constraint evaluators for the benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vis_ifeval.config import load_config
from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.evaluators.comp_eval import CompositionEvaluator
from vis_ifeval.evaluators.csp_eval import CSPEvaluator
from vis_ifeval.evaluators.label_eval import LabelEvaluator
from vis_ifeval.evaluators.logic_eval import LogicEvaluator
from vis_ifeval.evaluators.negative_eval import NegativeEvaluator
from vis_ifeval.evaluators.spatial_eval import SpatialEvaluator
from vis_ifeval.evaluators.text_eval import TextEvaluator
from vis_ifeval.utils.clip_utils import ClipConfig, ClipModelWrapper
from vis_ifeval.utils.ocr_backend import build_text_backend

if TYPE_CHECKING:
    from PIL import Image


class EvaluatorRegistry:
    """Central registry dispatching constraints to evaluators."""

    def __init__(self, evaluators: list[ConstraintEvaluator] | None = None) -> None:
        """Initialize the registry with evaluators.

        Args:
            evaluators: Optional list of evaluators. If None, creates default set.
        """
        if evaluators is None:
            cfg = load_config()
            backend = build_text_backend(cfg.ocr_backend)

            # For now we hardcode CLIP config here; later can be exposed via config.py.
            clip_wrapper = ClipModelWrapper(ClipConfig())

            self.evaluators: list[ConstraintEvaluator] = [
                TextEvaluator(backend),
                LabelEvaluator(backend),
                LogicEvaluator(backend),
                CSPEvaluator(backend),
                NegativeEvaluator(clip_wrapper),
                CompositionEvaluator(clip_wrapper),
                SpatialEvaluator(),  # Optional - will degrade gracefully
            ]
        else:
            self.evaluators = evaluators

    def score_constraint(
        self, image: "Image.Image", prompt: dict, constraint: dict
    ) -> float:
        """Score a constraint for a given image and prompt.

        Args:
            image: PIL Image to evaluate.
            prompt: Prompt dictionary containing the full prompt.
            constraint: Constraint dictionary to evaluate.

        Returns:
            Score in [0, 1] indicating how well the constraint is satisfied.

        Raises:
            ValueError: If no evaluator can handle the constraint type.
        """
        for evaluator in self.evaluators:
            if evaluator.can_handle(constraint):
                return evaluator.score(image, prompt, constraint)

        raise ValueError(
            f"No evaluator found for constraint type: {constraint.get('type', 'unknown')}"
        )

