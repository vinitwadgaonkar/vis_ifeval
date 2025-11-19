"""OCR backend abstraction for text extraction."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from PIL import Image


class TextBackend(ABC):
    """Abstract OCR backend."""

    @abstractmethod
    def extract_text(self, image: "Image.Image") -> str:
        """Return raw text from the given image.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        raise NotImplementedError


class TesseractBackend(TextBackend):
    """Simple OCR backend using pytesseract."""

    def __init__(self, lang: str = "eng") -> None:
        """Initialize Tesseract backend.

        Args:
            lang: Language code for Tesseract (default: "eng").
        """
        self.lang = lang

    def extract_text(self, image: "Image.Image") -> str:
        """Extract text using pytesseract.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        import pytesseract

        return pytesseract.image_to_string(image, lang=self.lang)


class AdvancedBackend(TextBackend):
    """Advanced OCR backend using Surya OCR or DeepSeek-OCR with fallback to Tesseract."""

    def __init__(self, model_name: str = "surya") -> None:
        """Initialize advanced backend.

        Args:
            model_name: Name of the advanced OCR model ("surya" or "deepseek").
                       Falls back to Tesseract if advanced model not available.
        """
        self.model_name = model_name.lower()
        self._backend = None
        self._fallback = None
        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initialize the OCR backend, with fallback to Tesseract."""
        import logging
        logger = logging.getLogger(__name__)

        if self.model_name == "surya":
            try:
                from surya.ocr import run_ocr
                from surya.model.detection.model import load_model as load_det_model
                from surya.model.recognition.model import load_model as load_rec_model
                from surya.model.detection.processor import load_processor as load_det_processor
                from surya.model.recognition.processor import load_processor as load_rec_processor

                # Load models once
                self._det_model = load_det_model()
                self._rec_model = load_rec_model()
                self._det_processor = load_det_processor()
                self._rec_processor = load_rec_processor()
                self._backend = "surya"
                return
            except ImportError:
                pass

        elif self.model_name == "deepseek":
            try:
                from transformers import AutoModelForCausalLM
                from deepseek_vl.models import VLChatProcessor
                import torch
                
                # Use the smaller 1.3b model for feasibility
                model_path = "deepseek-ai/deepseek-vl-1.3b-chat"
                logger.info(f"Loading DeepSeek-VL model from {model_path}...")
                
                self._vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
                self._tokenizer = self._vl_chat_processor.tokenizer

                # Determine device
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps" # DeepSeek might not fully support MPS out of box, but let's try
                    # If MPS fails, might need to fallback to CPU, but standard torch should work.
                    # Actually, for safety with custom kernels, let's stick to CPU if not CUDA for now
                    # unless we are sure.
                    # But let's try CPU first for compatibility if CUDA is missing.
                    self._device = "cpu" 
                
                if torch.cuda.is_available():
                    self._vl_gpt = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16
                    ).to(self._device).eval()
                else:
                    # CPU/MPS might need float32
                    self._vl_gpt = AutoModelForCausalLM.from_pretrained(
                        model_path, 
                        trust_remote_code=True
                    ).to(self._device).eval()
                
                self._backend = "deepseek"
                logger.info(f"DeepSeek-VL loaded successfully on {self._device}")
                return
            except ImportError as e:
                logger.warning(f"DeepSeek dependencies missing: {e}")
            except Exception as e:
                logger.warning(f"Failed to load DeepSeek model: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to Tesseract
        logger.warning(
            f"Advanced OCR backend '{self.model_name}' not available. "
            "Falling back to Tesseract. Install with: pip install surya-ocr or deepseek-vl"
        )
        self._fallback = TesseractBackend()
        self._backend = "tesseract"

    def extract_text(self, image: "Image.Image") -> str:
        """Extract text using advanced OCR or fallback to Tesseract.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        if self._backend == "surya":
            return self._extract_with_surya(image)
        elif self._backend == "deepseek":
            return self._extract_with_deepseek(image)
        elif self._fallback:
            return self._fallback.extract_text(image)
        else:
            # Should not reach here, but safety fallback
            return TesseractBackend().extract_text(image)

    def _extract_with_surya(self, image: "Image.Image") -> str:
        """Extract text using Surya OCR.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        from surya.ocr import run_ocr
        import numpy as np

        try:
            # Convert PIL Image to numpy array (RGB)
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_array = np.array(image)

            # Run OCR
            predictions = run_ocr(
                [img_array],
                [self._det_model],
                [self._rec_model],
                [self._det_processor],
                [self._rec_processor]
            )

            # Extract text from predictions
            if predictions and len(predictions) > 0:
                text_lines = []
                prediction = predictions[0]
                
                # Handle different possible response structures
                if hasattr(prediction, "text_lines"):
                    for line in prediction.text_lines:
                        if hasattr(line, "text") and line.text:
                            text_lines.append(line.text)
                elif hasattr(prediction, "text"):
                    # If prediction has direct text attribute
                    text_lines.append(prediction.text)
                elif isinstance(prediction, str):
                    text_lines.append(prediction)
                
                return "\n".join(text_lines) if text_lines else ""
            else:
                return ""
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Surya OCR extraction failed: {e}. Falling back to Tesseract.")
            # Fallback to Tesseract on error
            return TesseractBackend().extract_text(image)

    def _extract_with_deepseek(self, image: "Image.Image") -> str:
        """Extract text using DeepSeek-VL.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        from deepseek_vl.utils.io import load_pil_images
        import tempfile
        import torch
        
        # Create temp file for the image because load_pil_images expects it?
        # Actually load_pil_images helper might be needed, or we can construct input manually.
        # But let's stick to standard usage.
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.convert("RGB").save(tmp.name)
            tmp_path = tmp.name

        try:
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>Extract all text from this image verbatim, preserving line breaks.",
                    "images": [tmp_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            pil_images = load_pil_images(conversation)
            prepare_inputs = self._vl_chat_processor(
                conversations=[conversation],
                images=pil_images,
                force_batchify=True
            ).to(self._device)
            
            # Generate
            inputs_embeds = self._vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            with torch.no_grad():
                outputs = self._vl_gpt.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self._tokenizer.eos_token_id,
                    bos_token_id=self._tokenizer.bos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                    max_new_tokens=512,
                    do_sample=False,
                    use_cache=True
                )
                
            answer = self._tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DeepSeek OCR extraction failed: {e}. Falling back to Tesseract.")
            return TesseractBackend().extract_text(image)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def build_text_backend(name: str) -> TextBackend:
    """Factory to select OCR backend by name.

    Args:
        name: Backend name ("tesseract", "default", "advanced", "surya", "deepseek").

    Returns:
        TextBackend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    name = name.lower()
    if name in ("tesseract", "default"):
        return TesseractBackend()
    elif name in ("advanced", "surya", "deepseek"):
        return AdvancedBackend(model_name=name)
    else:
        raise ValueError(f"Unknown OCR backend: {name}")
