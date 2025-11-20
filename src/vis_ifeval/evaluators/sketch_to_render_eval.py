"""Sketch to render evaluator using edge detection and SSIM."""

import logging
import math
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.evaluators.base import ConstraintEvaluator
from vis_ifeval.utils.clip_utils import ClipModelWrapper

logger = logging.getLogger(__name__)


class SketchToRenderEvaluator(ConstraintEvaluator):
    """
    Evaluator for sketch-to-render fidelity.
    
    Uses:
    - HED or Canny edge detection
    - SSIM (structural similarity)
    - CLIP for prompt adherence
    - Texture analyzer (Gabor filters or learned)
    """

    def __init__(self, clip: ClipModelWrapper | None = None) -> None:
        """Initialize SketchToRenderEvaluator.
        
        Args:
            clip: Optional ClipModelWrapper for prompt adherence.
        """
        self.clip = clip
        self._warned_clip = False

    def can_handle(self, constraint: Dict[str, Any]) -> bool:
        """Check if constraint type is 'sketch_to_render'."""
        return constraint.get("type") == "sketch_to_render"

    def _extract_edges_hed(self, image: "Image.Image") -> Any:
        """Extract edges using HED (Holistically-Nested Edge Detection).
        
        Returns:
            Edge map as numpy array.
        """
        try:
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV format
            img_array = np.array(image.convert("RGB"))
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # HED requires a pre-trained model, but we can use Canny as fallback
            # For now, use Canny edge detection
            edges = cv2.Canny(img_gray, 50, 150)
            return edges
        except ImportError:
            logger.debug("OpenCV not available for edge detection")
            return None
        except Exception as e:
            logger.debug(f"Edge extraction failed: {e}")
            return None

    def _extract_edges_canny(self, image: "Image.Image") -> Any:
        """Extract edges using Canny edge detection.
        
        Returns:
            Edge map as numpy array.
        """
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image.convert("RGB"))
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            return edges
        except ImportError:
            logger.debug("OpenCV not available for edge detection")
            return None
        except Exception as e:
            logger.debug(f"Canny edge extraction failed: {e}")
            return None

    def _compute_ssim(self, img1: "Image.Image", img2: "Image.Image") -> float:
        """Compute Structural Similarity Index (SSIM) between two images.
        
        Args:
            img1: First image.
            img2: Second image.
            
        Returns:
            SSIM score in [0, 1].
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            import numpy as np
            
            # Resize images to same size
            img1_resized = img1.resize((256, 256))
            img2_resized = img2.resize((256, 256))
            
            # Convert to grayscale for SSIM
            img1_gray = np.array(img1_resized.convert("L"))
            img2_gray = np.array(img2_resized.convert("L"))
            
            # Compute SSIM
            score = ssim(img1_gray, img2_gray, data_range=255)
            return float(score)
        except ImportError:
            logger.debug("scikit-image not available for SSIM")
            return 0.0
        except Exception as e:
            logger.debug(f"SSIM computation failed: {e}")
            return 0.0

    def _compute_edge_alignment(
        self, sketch_edges: Any, render_edges: Any
    ) -> float:
        """Compute edge alignment score between sketch and render.
        
        Args:
            sketch_edges: Edge map from sketch.
            render_edges: Edge map from render.
            
        Returns:
            Edge alignment score in [0, 1].
        """
        if sketch_edges is None or render_edges is None:
            return 0.0

        try:
            import numpy as np
            import cv2
            
            # Resize to same size
            h, w = sketch_edges.shape
            render_edges_resized = cv2.resize(render_edges, (w, h))
            
            # Normalize
            sketch_norm = sketch_edges.astype(float) / 255.0
            render_norm = render_edges_resized.astype(float) / 255.0
            
            # Compute intersection over union of edges
            intersection = np.logical_and(sketch_norm > 0.5, render_norm > 0.5).sum()
            union = np.logical_or(sketch_norm > 0.5, render_norm > 0.5).sum()
            
            if union == 0:
                return 0.0
            
            iou = intersection / union
            return float(iou)
        except Exception as e:
            logger.debug(f"Edge alignment computation failed: {e}")
            return 0.0

    def _compute_texture_richness(self, image: "Image.Image") -> float:
        """Compute texture richness using variance of gradients.
        
        Args:
            image: PIL Image to analyze.
            
        Returns:
            Texture richness score in [0, 1].
        """
        try:
            import cv2
            import numpy as np
            
            img_array = np.array(image.convert("RGB"))
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Compute gradients
            grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Compute variance as measure of texture richness
            variance = np.var(gradient_magnitude)
            
            # Normalize to [0, 1] (assuming max variance around 10000)
            normalized = min(1.0, variance / 10000.0)
            return float(normalized)
        except Exception as e:
            logger.debug(f"Texture richness computation failed: {e}")
            return 0.5  # Neutral score

    def _check_prompt_adherence(
        self, image: "Image.Image", prompt_text: str
    ) -> float:
        """Check prompt adherence using CLIP.
        
        Args:
            image: PIL Image to check.
            prompt_text: Text prompt.
            
        Returns:
            CLIP similarity score in [0, 1].
        """
        if self.clip is None or not self.clip.enabled:
            if not self._warned_clip:
                logger.debug("CLIP not available for prompt adherence check")
                self._warned_clip = True
            return 0.5  # Neutral score

        try:
            sims = self.clip.image_text_similarities(image, [prompt_text])
            if sims and len(sims) > 0:
                # Normalize CLIP score (typically in [-1, 1]) to [0, 1]
                normalized = (sims[0] + 1.0) / 2.0
                return float(max(0.0, min(1.0, normalized)))
        except Exception as e:
            logger.debug(f"Prompt adherence check failed: {e}")

        return 0.5

    def score(
        self, image: "Image.Image", prompt: Dict[str, Any], constraint: Dict[str, Any]
    ) -> float:
        """Score sketch-to-render constraint.
        
        This evaluator expects:
        - constraint["type"] == "sketch_to_render"
        - constraint["sketch_image"]: Reference sketch image (PIL Image)
        - constraint["style"]: Optional style description
        
        Args:
            image: PIL Image to evaluate (rendered image).
            prompt: Full prompt dictionary.
            constraint: Constraint dictionary.
            
        Returns:
            Score in [0, 1] indicating sketch-to-render fidelity.
        """
        sketch_image = constraint.get("sketch_image")
        if sketch_image is None:
            logger.warning("No sketch image provided for sketch-to-render evaluation")
            return 0.0

        style = constraint.get("style", "")
        prompt_text = prompt.get("prompt", "")

        scores = []

        # 1. Structural fidelity (SSIM)
        ssim_score = self._compute_ssim(sketch_image, image)
        scores.append(ssim_score)

        # 2. Edge alignment
        sketch_edges = self._extract_edges_canny(sketch_image)
        render_edges = self._extract_edges_canny(image)
        edge_score = self._compute_edge_alignment(sketch_edges, render_edges)
        scores.append(edge_score)

        # 3. Prompt adherence (using CLIP)
        if prompt_text:
            prompt_score = self._check_prompt_adherence(image, prompt_text)
            scores.append(prompt_score)

        # 4. Texture richness (detail level)
        # Compare texture richness between sketch and render
        sketch_texture = self._compute_texture_richness(sketch_image)
        render_texture = self._compute_texture_richness(image)
        # Render should have more texture than sketch
        texture_score = min(1.0, render_texture / max(0.1, sketch_texture))
        scores.append(texture_score)

        # 5. Style match (if style specified)
        if style and self.clip and self.clip.enabled:
            style_score = self._check_prompt_adherence(image, style)
            scores.append(style_score)

        # Combine scores (weighted average)
        if scores:
            # Adjust weights: SSIM and prompt adherence are most important
            # Edge alignment can be lower if sketch and render are independently generated
            # Use dynamic weights based on number of scores
            if len(scores) == 4:  # SSIM, edge, prompt, texture
                weights = [0.35, 0.15, 0.35, 0.15]
            elif len(scores) == 5:  # SSIM, edge, prompt, texture, style
                weights = [0.30, 0.10, 0.30, 0.15, 0.15]
            else:
                # Equal weights as fallback
                weights = [1.0 / len(scores)] * len(scores)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            final_score = sum(s * w for s, w in zip(scores, weights))
            return float(max(0.0, min(1.0, final_score)))
        else:
            return 0.0

