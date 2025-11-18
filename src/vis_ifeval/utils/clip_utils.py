"""CLIP utilities for image-text similarity."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ClipConfig:
    """Configuration for CLIP model."""

    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    device: str = "cpu"  # later: "cuda" if available


class ClipModelWrapper:
    """
    Thin wrapper around an OpenCLIP model.

    - Handles lazy loading.
    - Provides encode_image / encode_text utilities.
    - If open_clip is unavailable, wrapper is disabled and callers must degrade gracefully.
    """

    def __init__(self, cfg: Optional[ClipConfig] = None) -> None:
        """Initialize CLIP wrapper.

        Args:
            cfg: Optional CLIP configuration. If None, uses defaults.
        """
        self.cfg = cfg or ClipConfig()
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.enabled = False
        self.device = "cpu"

        try:
            import torch
            import open_clip

            device = self.cfg.device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning(
                    "CLIP device set to cuda but CUDA not available, falling back to cpu"
                )
                device = "cpu"

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.cfg.model_name,
                pretrained=self.cfg.pretrained,
                device=device,
            )
            tokenizer = open_clip.get_tokenizer(self.cfg.model_name)

            self.model = model
            self.preprocess = preprocess
            self.tokenizer = tokenizer
            self.device = device
            self.enabled = True
            logger.info(
                "Loaded CLIP model %s (%s) on %s",
                self.cfg.model_name,
                self.cfg.pretrained,
                device,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize CLIP model, disabling CLIP-based evaluators: %s", exc
            )
            self.enabled = False
            self.model = None
            self.preprocess = None
            self.tokenizer = None
            self.device = "cpu"

    def encode_image(self, image: "Image.Image") -> Optional[np.ndarray]:
        """Return normalized image embedding, or None if disabled.

        Args:
            image: PIL Image to encode.

        Returns:
            Normalized image embedding as numpy array, or None if disabled.
        """
        if not self.enabled or self.model is None or self.preprocess is None:
            return None

        import torch

        img = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def encode_texts(self, texts: List[str]) -> Optional[np.ndarray]:
        """Return normalized text embeddings for a list of strings, or None if disabled.

        Args:
            texts: List of text strings to encode.

        Returns:
            Normalized text embeddings as numpy array, or None if disabled.
        """
        if not self.enabled or self.model is None or self.tokenizer is None:
            return None

        import torch

        tokens = self.tokenizer(texts)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized vectors, in [-1, 1].

        Args:
            a: First normalized vector.
            b: Second normalized vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

    def image_text_similarities(
        self,
        image: "Image.Image",
        texts: List[str],
    ) -> Optional[List[float]]:
        """Compute cosine similarities between image and each text.

        Args:
            image: PIL Image to compare.
            texts: List of text strings to compare against.

        Returns:
            List of cosine similarities, or None if disabled.
        """
        img_emb = self.encode_image(image)
        if img_emb is None:
            return None
        txt_embs = self.encode_texts(texts)
        if txt_embs is None:
            return None
        sims = [self.cosine_sim(img_emb, t) for t in txt_embs]
        return sims
