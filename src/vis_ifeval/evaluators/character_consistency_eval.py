"""Character consistency evaluator using face recognition and object detection."""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.clip_utils import ClipModelWrapper

logger = logging.getLogger(__name__)


class CharacterConsistencyEvaluator(ConstraintEvaluator):
    """
    Evaluator for character consistency across multiple images.
    
    Uses:
    - InsightFace or ArcFace for face recognition
    - YOLOv8 or GroundingDINO for character detection
    - CLIP for attribute verification
    - Face embedding comparison (cosine similarity)
    """

    def __init__(self, clip: ClipModelWrapper | None = None) -> None:
        """Initialize CharacterConsistencyEvaluator.
        
        Args:
            clip: Optional ClipModelWrapper for attribute verification.
        """
        self.clip = clip
        self.face_model = None
        self.detector = None
        self._warned_face = False
        self._warned_detector = False
        self._try_load_face_model()
        self._try_load_detector()

    def _try_load_face_model(self) -> None:
        """Try to load InsightFace model for face recognition."""
        try:
            import insightface
            import onnxruntime
            
            # Try to load InsightFace model
            # InsightFace can auto-download models
            try:
                self.face_model = insightface.app.FaceAnalysis(
                    name='buffalo_l',  # or 'buffalo_s' for smaller model
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.face_model.prepare(ctx_id=0, det_size=(640, 640))
                logger.info("InsightFace model loaded successfully")
            except Exception as e:
                logger.debug(f"InsightFace model loading failed: {e}")
                self.face_model = None
        except ImportError:
            logger.debug("InsightFace not available. Install with: pip install insightface")

    def _try_load_detector(self) -> None:
        """Try to load YOLOv8 or GroundingDINO for character detection."""
        # Try YOLOv8 first
        try:
            from ultralytics import YOLO
            self.detector = YOLO('yolov8n.pt')  # nano model for speed
            self.detector_type = 'yolo'
            logger.info("YOLOv8 detector loaded successfully")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"YOLOv8 loading failed: {e}")

        # Fallback to GroundingDINO
        try:
            from vis_ifeval.evaluators.spatial_eval import SpatialEvaluator
            spatial_eval = SpatialEvaluator()
            if spatial_eval.detector:
                self.detector = spatial_eval.model
                self.detector_type = 'groundingdino'
                logger.info("Using GroundingDINO from SpatialEvaluator")
                return
        except Exception:
            pass

        logger.debug("No object detector available for character detection")

    def can_handle(self, constraint: Dict[str, Any]) -> bool:
        """Check if constraint type is 'character_consistency'."""
        return constraint.get("type") == "character_consistency"

    def _extract_face_embeddings(self, image: "Image.Image") -> List[Tuple[Any, float]]:
        """Extract face embeddings from image.
        
        Returns:
            List of (embedding, confidence) tuples.
        """
        if self.face_model is None:
            return []

        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            faces = self.face_model.get(img_array)
            
            embeddings = []
            for face in faces:
                if face.embedding is not None:
                    embeddings.append((face.embedding, face.det_score))
            return embeddings
        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
            return []

    def _detect_characters(self, image: "Image.Image", character_description: str) -> List[Tuple[float, float, float, float]]:
        """Detect characters in image using object detection.
        
        Args:
            image: PIL Image to detect characters in.
            character_description: Text description of character.
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2) tuples.
        """
        if self.detector is None:
            return []

        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            
            if self.detector_type == 'yolo':
                # YOLOv8 detection
                results = self.detector(img_array, verbose=False)
                bboxes = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                return bboxes
            elif self.detector_type == 'groundingdino':
                # GroundingDINO detection
                from groundingdino.util.inference import predict
                import torch
                import torchvision.transforms as transforms
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ])
                img_tensor = transform(img_array)
                
                device = next(self.detector.parameters()).device
                text_prompt = f"{character_description}. ."
                boxes, _, _ = predict(
                    model=self.detector,
                    image=img_tensor,
                    caption=text_prompt,
                    box_threshold=0.2,
                    text_threshold=0.2,
                    device=device
                )
                
                if boxes is None or len(boxes) == 0:
                    return []
                
                bboxes = []
                h, w = img_array.shape[:2]
                for box in boxes:
                    cx, cy, width, height = box
                    x1 = (cx - width / 2) * w
                    y1 = (cy - height / 2) * h
                    x2 = (cx + width / 2) * w
                    y2 = (cy + height / 2) * h
                    bboxes.append((float(x1), float(y1), float(x2), float(y2)))
                return bboxes
        except Exception as e:
            logger.debug(f"Character detection failed: {e}")
            return []

        return []

    def _cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            import numpy as np
            vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
            vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
            return float(np.dot(vec1_norm, vec2_norm))
        except Exception:
            return 0.0

    def _check_attribute_consistency(
        self, image: "Image.Image", attributes: List[str]
    ) -> float:
        """Check if attributes are consistent using CLIP.
        
        Args:
            image: PIL Image to check.
            attributes: List of attribute descriptions.
            
        Returns:
            Average CLIP similarity score for attributes.
        """
        if self.clip is None or not self.clip.enabled:
            return 0.5  # Neutral score if CLIP unavailable

        try:
            sims = self.clip.image_text_similarities(image, attributes)
            if sims and len(sims) > 0:
                return float(sum(sims) / len(sims))
        except Exception as e:
            logger.debug(f"Attribute consistency check failed: {e}")

        return 0.5

    def score(
        self, image: "Image.Image", prompt: Dict[str, Any], constraint: Dict[str, Any]
    ) -> float:
        """Score character consistency constraint.
        
        This evaluator expects:
        - constraint["type"] == "character_consistency"
        - constraint["reference_images"]: List of reference images (PIL Images)
        - constraint["character_description"]: Optional text description
        - constraint["attributes"]: Optional list of attributes to check
        
        Args:
            image: PIL Image to evaluate (current image).
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary.
            
        Returns:
            Score in [0, 1] indicating character consistency.
        """
        reference_images = constraint.get("reference_images", [])
        if not reference_images:
            logger.warning("No reference images provided for character consistency")
            return 0.0

        character_description = constraint.get("character_description", "person")
        attributes = constraint.get("attributes", [])

        scores = []

        # 1. Face similarity (if faces detected)
        if self.face_model is not None:
            current_faces = self._extract_face_embeddings(image)
            if current_faces:
                best_face_sim = 0.0
                for ref_img in reference_images:
                    ref_faces = self._extract_face_embeddings(ref_img)
                    for curr_emb, curr_conf in current_faces:
                        for ref_emb, ref_conf in ref_faces:
                            sim = self._cosine_similarity(curr_emb, ref_emb)
                            # Weight by detection confidence
                            weighted_sim = sim * min(curr_conf, ref_conf)
                            best_face_sim = max(best_face_sim, weighted_sim)
                if best_face_sim > 0:
                    scores.append(best_face_sim)
            elif not self._warned_face:
                logger.debug("No faces detected in images for character consistency")
                self._warned_face = True

        # 2. Character detection consistency
        if self.detector is not None:
            current_chars = self._detect_characters(image, character_description)
            if current_chars:
                # Check if character is detected in reference images too
                ref_detections = 0
                for ref_img in reference_images:
                    ref_chars = self._detect_characters(ref_img, character_description)
                    if ref_chars:
                        ref_detections += 1
                if ref_detections > 0:
                    # Score based on detection consistency
                    detection_score = min(1.0, ref_detections / len(reference_images))
                    scores.append(detection_score)

        # 3. Attribute consistency (using CLIP)
        if attributes and self.clip and self.clip.enabled:
            current_attr_score = self._check_attribute_consistency(image, attributes)
            ref_attr_scores = []
            for ref_img in reference_images:
                ref_score = self._check_attribute_consistency(ref_img, attributes)
                ref_attr_scores.append(ref_score)
            
            if ref_attr_scores:
                avg_ref_score = sum(ref_attr_scores) / len(ref_attr_scores)
                # Compare current image attributes to reference average
                attr_consistency = 1.0 - abs(current_attr_score - avg_ref_score)
                scores.append(max(0.0, attr_consistency))

        # Combine scores (average if multiple metrics available)
        if scores:
            final_score = sum(scores) / len(scores)
            return float(max(0.0, min(1.0, final_score)))
        else:
            # No metrics available - return neutral score
            if not self._warned_detector:
                logger.warning(
                    "CharacterConsistencyEvaluator: No face recognition or object detection available. "
                    "Install InsightFace and/or YOLOv8 for character consistency evaluation."
                )
                self._warned_detector = True
            return 0.5  # Neutral score when no detection available

