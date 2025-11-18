"""Stable Diffusion 3 model implementation."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from PIL import Image

from vis_ifeval.models.base_model import ImageModel


class SD3Model(ImageModel):
    """Stable Diffusion 3 model for image generation."""

    def __init__(
        self,
        name: str = "sd3",
        model_id: str = "stabilityai/stable-diffusion-3-medium",
        device: str = "cuda",
        guidance_scale: float = 7.0,
        num_inference_steps: int = 28,
    ) -> None:
        """Initialize SD3 model.

        Args:
            name: Model name.
            model_id: HuggingFace model ID.
            device: Device to run on ("cuda" or "cpu").
            guidance_scale: Guidance scale for generation.
            num_inference_steps: Number of inference steps.
        """
        super().__init__(
            name=name,
            config={
                "model_id": model_id,
                "device": device,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            },
        )
        self.model_id = model_id
        self.device = device
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.pipe = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load the SD3 pipeline."""
        try:
            import torch
            from diffusers import StableDiffusion3Pipeline

            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)
        except ImportError as e:
            raise ImportError(
                "SD3 model requires torch and diffusers. "
                "Install with: pip install torch diffusers transformers"
            ) from e

    def generate(self, prompt: str, seed: Optional[int] = None) -> "Image.Image":
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt describing the desired image.
            seed: Optional random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        if self.pipe is None:
            self._load_model()

        import torch

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        out = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        )
        img = out.images[0]
        return img

