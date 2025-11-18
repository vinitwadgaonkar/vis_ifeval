"""Spatial relation evaluator using object detection."""

import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator

logger = logging.getLogger(__name__)


class SpatialEvaluator(ConstraintEvaluator):
    """
    Evaluator for spatial relationship constraints.

    Uses GroundingDINO for object detection to find objects and check spatial relationships.
    Falls back gracefully if GroundingDINO is not available.
    """

    def __init__(self) -> None:
        """Initialize SpatialEvaluator."""
        self._warned_disabled = False
        self.detector = None
        self.model = None
        self.tokenizer = None
        self._try_load_detector()

    def _try_load_detector(self) -> None:
        """Try to load GroundingDINO object detection model."""
        try:
            from groundingdino.util.inference import load_model, load_image, predict, annotate
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
            import torch

            # Try to load GroundingDINO model
            # Default config and checkpoint paths
            config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            checkpoint_path = "groundingdino_swint_ogc.pth"

            # Try alternative paths or environment variables
            import os
            config_file = os.getenv("GROUNDINGDINO_CONFIG", config_file)
            checkpoint_path = os.getenv("GROUNDINGDINO_CHECKPOINT", checkpoint_path)

            try:
                self.model = load_model(config_file, checkpoint_path)
                self.detector = "groundingdino"
                logger.info("GroundingDINO model loaded successfully")
            except (FileNotFoundError, OSError) as e:
                logger.debug(f"GroundingDINO model files not found: {e}")
                logger.debug("Install GroundingDINO and download model weights for spatial evaluation")
        except ImportError:
            logger.debug("GroundingDINO not available for spatial evaluation. Install with: pip install groundingdino-py")

    def can_handle(self, constraint: Dict[str, Any]) -> bool:
        """Check if constraint type is 'spatial'."""
        return constraint.get("type") == "spatial"

    def _check_spatial_relation(
        self,
        bbox_a: tuple[float, float, float, float],
        bbox_b: tuple[float, float, float, float],
        relation: str,
    ) -> bool:
        """Check if bbox_a has the specified relation to bbox_b.

        Args:
            bbox_a: Bounding box (x1, y1, x2, y2) for object A.
            bbox_b: Bounding box (x1, y1, x2, y2) for object B.
            relation: Spatial relation ("left_of", "right_of", "above", "below").

        Returns:
            True if relation is satisfied.
        """
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        # Compute centers
        center_a = ((x1_a + x2_a) / 2, (y1_a + y2_a) / 2)
        center_b = ((x1_b + x2_b) / 2, (y1_b + y2_b) / 2)

        if relation == "left_of":
            return center_a[0] < center_b[0]
        elif relation == "right_of":
            return center_a[0] > center_b[0]
        elif relation == "above":
            return center_a[1] < center_b[1]
        elif relation == "below":
            return center_a[1] > center_b[1]
        else:
            logger.warning(f"Unknown spatial relation: {relation}")
            return False

    def score(
        self, image: "Image.Image", prompt: Dict[str, Any], constraint: Dict[str, Any]
    ) -> float:
        """Score spatial constraint.

        Args:
            image: PIL Image to evaluate.
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary with spatial relation.

        Returns:
            Score in [0, 1] indicating if spatial relation is satisfied.
        """
        if self.detector is None:
            if not self._warned_disabled:
                logger.warning(
                    "SpatialEvaluator: Object detection not available. "
                    "Install GroundingDINO for spatial evaluation. Returning 0.0."
                )
                self._warned_disabled = True
            return 0.0

        # Extract constraint info
        object_a = constraint.get("object_a") or constraint.get("object", "")
        object_b = constraint.get("object_b") or constraint.get("target", "")
        relation = constraint.get("relation", "")

        if not object_a or not object_b or not relation:
            logger.warning("Spatial constraint missing required fields")
            return 0.0

        try:
            if self.detector != "groundingdino" or self.model is None:
                if not self._warned_disabled:
                    logger.warning(
                        "SpatialEvaluator: GroundingDINO not available. "
                        "Install with: pip install groundingdino-py. Returning 0.0."
                    )
                    self._warned_disabled = True
                return 0.0

            # Detect objects using GroundingDINO
            bboxes_a = self._detect_objects(image, object_a)
            bboxes_b = self._detect_objects(image, object_b)

            if not bboxes_a or not bboxes_b:
                logger.debug(f"Could not detect objects: '{object_a}' or '{object_b}'")
                return 0.0

            # Check spatial relation for best matches
            # Use the first detected instance of each object
            bbox_a = bboxes_a[0]
            bbox_b = bboxes_b[0]

            satisfied = self._check_spatial_relation(bbox_a, bbox_b, relation)
            return 1.0 if satisfied else 0.0

        except Exception as e:
            logger.warning(f"Spatial evaluation failed: {e}")
            return 0.0

    def _detect_objects(self, image: "Image.Image", object_description: str) -> list[tuple[float, float, float, float]]:
        """Detect objects in image using GroundingDINO.

        Args:
            image: PIL Image to detect objects in.
            object_description: Text description of object to detect.

        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples.
        """
        from groundingdino.util.inference import predict
        import torch
        import numpy as np
        from PIL import Image

        # Convert PIL to numpy array
        img_array = np.array(image.convert("RGB"))
        
        # Set up text prompt
        text_prompt = f"{object_description}. ."
        box_threshold = 0.3
        text_threshold = 0.25

        try:
            # Run detection
            boxes, logits, phrases = predict(
                model=self.model,
                image=img_array,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            if boxes is None or len(boxes) == 0:
                return []

            # Convert boxes to (x1, y1, x2, y2) format
            # GroundingDINO returns boxes in normalized format [cx, cy, w, h]
            bboxes = []
            h, w = img_array.shape[:2]
            
            for box in boxes:
                # Convert from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel coordinates
                cx, cy, width, height = box
                x1 = (cx - width / 2) * w
                y1 = (cy - height / 2) * h
                x2 = (cx + width / 2) * w
                y2 = (cy + height / 2) * h
                bboxes.append((float(x1), float(y1), float(x2), float(y2)))

            return bboxes

        except Exception as e:
            logger.debug(f"Object detection failed for '{object_description}': {e}")
            return []

