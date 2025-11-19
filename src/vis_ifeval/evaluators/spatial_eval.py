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
            import os

            # Find config file - try package location first, then environment variable, then relative path
            try:
                import groundingdino
                package_dir = os.path.dirname(groundingdino.__file__)
                config_file = os.path.join(package_dir, "config", "GroundingDINO_SwinT_OGC.py")
                if not os.path.exists(config_file):
                    config_file = None
            except:
                config_file = None

            # Try environment variable
            if not config_file:
                config_file = os.getenv("GROUNDINGDINO_CONFIG")

            # Try relative path as fallback
            if not config_file or not os.path.exists(config_file):
                config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

            # Find checkpoint file - try relative path, then environment variable
            checkpoint_path = os.getenv("GROUNDINGDINO_CHECKPOINT")
            if not checkpoint_path:
                # Try common locations
                possible_paths = [
                    "weights/groundingdino_swint_ogc.pth",
                    "groundingdino_swint_ogc.pth",
                    os.path.expanduser("~/weights/groundingdino_swint_ogc.pth"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        checkpoint_path = path
                        break

            if not checkpoint_path:
                logger.debug("GroundingDINO checkpoint not found. Download from: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
                return

            if not os.path.exists(config_file):
                logger.debug(f"GroundingDINO config file not found at: {config_file}")
                return

            try:
                # Load model (use CPU if CUDA not available)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = load_model(config_file, checkpoint_path, device=device)
                self.model.eval()  # Set to evaluation mode
                self.detector = "groundingdino"
                logger.info(f"GroundingDINO model loaded successfully (device: {device})")
            except (FileNotFoundError, OSError) as e:
                logger.debug(f"GroundingDINO model loading failed: {e}")
                logger.debug("Ensure model weights are downloaded and paths are correct")
            except Exception as e:
                logger.debug(f"GroundingDINO model loading error: {e}")
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
        from groundingdino.util.inference import predict, load_image
        import torch
        import numpy as np
        from PIL import Image

        # Convert PIL image to format expected by GroundingDINO
        # load_image returns (image, image_source) tuple where image is a torch tensor
        try:
            # Save PIL image temporarily and use load_image, or convert directly
            # GroundingDINO's load_image expects a file path, so we'll convert manually
            img_array = np.array(image.convert("RGB"))
            # Convert numpy array to torch tensor and normalize
            import torchvision.transforms as transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])
            img_tensor = transform(img_array)
            
        except Exception:
            # Fallback: use numpy array directly (may need adjustment)
            img_array = np.array(image.convert("RGB"))
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
        
        # Set up text prompt (GroundingDINO expects format: "object. .")
        text_prompt = f"{object_description}. ."
        # Lower thresholds for better detection (can be adjusted)
        box_threshold = 0.2
        text_threshold = 0.2

        try:
            # Ensure model is on correct device
            device = next(self.model.parameters()).device
            # Ensure device is CPU if CUDA not available
            if not torch.cuda.is_available():
                device = "cpu"
            
            # Run detection - explicitly pass device to predict function
            self.model = self.model.to(device)
            boxes, logits, phrases = predict(
                model=self.model,
                image=img_tensor,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=device  # Explicitly pass device to avoid CUDA errors
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

