"""Base class for constraint evaluators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


class ConstraintEvaluator(ABC):
    """Abstract base class for constraint evaluators."""

    @abstractmethod
    def can_handle(self, constraint: dict) -> bool:
        """Check if this evaluator can handle the given constraint.

        Args:
            constraint: Constraint dictionary.

        Returns:
            True if this evaluator can handle the constraint, False otherwise.
        """
        pass

    @abstractmethod
    def score(
        self, image: "Image.Image", prompt: dict, constraint: dict
    ) -> float:
        """Score how well an image satisfies a constraint.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary to evaluate.

        Returns:
            Score in [0, 1] where 1.0 means fully satisfied, 0.0 means not satisfied.
        """
        pass

