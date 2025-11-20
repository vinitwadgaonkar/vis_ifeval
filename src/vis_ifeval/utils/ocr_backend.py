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
                from transformers import AutoModel, AutoTokenizer
                import torch
                import tempfile
                import os
                import sys
                
                # Use the latest DeepSeek-OCR model
                model_path = "deepseek-ai/DeepSeek-OCR"
                logger.info(f"Loading DeepSeek-OCR model from {model_path}...")
                
                # Determine device
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Patch the import issue with LlamaFlashAttention2 if needed
                # The model code tries to import this but it may not exist in all transformers versions
                try:
                    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
                except ImportError:
                    # Create a dummy class to avoid import errors in model code
                    class LlamaFlashAttention2:
                        pass
                    import transformers.models.llama.modeling_llama as llama_module
                    if not hasattr(llama_module, 'LlamaFlashAttention2'):
                        llama_module.LlamaFlashAttention2 = LlamaFlashAttention2
                
                # Patch DynamicCache to add compatibility methods
                try:
                    from transformers.cache_utils import DynamicCache
                    if not hasattr(DynamicCache, 'seen_tokens'):
                        # Add seen_tokens property to DynamicCache
                        def _get_seen_tokens(self):
                            # Calculate seen tokens from cache
                            if hasattr(self, 'key_cache') and self.key_cache:
                                return self.key_cache[0].shape[2] if len(self.key_cache) > 0 else 0
                            return 0
                        DynamicCache.seen_tokens = property(_get_seen_tokens)
                    
                    # Patch get_max_length to use get_seq_length (newer transformers API)
                    if not hasattr(DynamicCache, 'get_max_length'):
                        def _get_max_length(self):
                            return self.get_seq_length() if hasattr(self, 'get_seq_length') else 0
                        DynamicCache.get_max_length = _get_max_length
                    
                    # Patch get_usable_length (newer transformers API)
                    if not hasattr(DynamicCache, 'get_usable_length'):
                        def _get_usable_length(self, seq_length):
                            # Return the usable length from cache
                            if hasattr(self, 'get_seq_length'):
                                return self.get_seq_length()
                            return 0
                        DynamicCache.get_usable_length = _get_usable_length
                except ImportError:
                    pass
                
                # Load tokenizer and model
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=True
                )
                
                if torch.cuda.is_available():
                    # Use default attention (flash_attention_2 requires additional setup)
                    # The model will work fine with default attention
                    self._model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_safetensors=True
                    ).eval().cuda().to(torch.bfloat16)
                else:
                    # CPU mode
                    self._model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        use_safetensors=True
                    ).eval()
                
                # Create temp directory for image processing
                self._temp_dir = tempfile.mkdtemp()
                
                self._backend = "deepseek"
                logger.info(f"DeepSeek-OCR loaded successfully on {self._device}")
                return
            except ImportError as e:
                logger.warning(f"DeepSeek dependencies missing: {e}")
            except Exception as e:
                logger.warning(f"Failed to load DeepSeek-OCR model: {e}")
                import traceback
                traceback.print_exc()

        # Fallback to Tesseract
        logger.warning(
            f"Advanced OCR backend '{self.model_name}' not available. "
            "Falling back to Tesseract. Install with: pip install surya-ocr or transformers"
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
        """Extract text using DeepSeek-OCR.

        Args:
            image: PIL Image to extract text from.

        Returns:
            Extracted text as a string.
        """
        import tempfile
        import os
        import logging
        logger = logging.getLogger(__name__)
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self._temp_dir) as tmp:
            image.convert("RGB").save(tmp.name)
            tmp_path = tmp.name

        try:
            # Use DeepSeek-OCR's infer method with OCR prompt
            # The model expects: prompt, image_file, output_path, and other parameters
            prompt = "<image>\nFree OCR. "
            
            # Capture stdout since infer might print the result
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            try:
                # Call the model's infer method
                # Parameters: base_size=1024, image_size=640, crop_mode=True for "Gundam" mode
                # For faster inference, we can use smaller sizes: base_size=640, image_size=640, crop_mode=False
                result = self._model.infer(
                    self._tokenizer,
                    prompt=prompt,
                    image_file=tmp_path,
                    output_path=self._temp_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                    test_compress=False
                )
            finally:
                sys.stdout = old_stdout
            
            # Get captured output
            captured_text = captured_output.getvalue()
            
            # The infer method may return different types depending on the model
            # It might also print to stdout, so check both
            extracted_text = ""
            
            # First, check if result is a string or has text
            if isinstance(result, str) and result.strip():
                extracted_text = result.strip()
            elif result is not None:
                # Try to extract from result object
                if isinstance(result, (list, tuple)) and len(result) > 0:
                    first = result[0]
                    if isinstance(first, str):
                        extracted_text = first.strip()
                    elif hasattr(first, 'text'):
                        extracted_text = str(first.text).strip()
                elif hasattr(result, 'text'):
                    extracted_text = str(result.text).strip()
                elif hasattr(result, '__dict__'):
                    # Try to find text in the object's attributes
                    for attr in ['text', 'content', 'output', 'result']:
                        if hasattr(result, attr):
                            val = getattr(result, attr)
                            if isinstance(val, str) and val.strip():
                                extracted_text = val.strip()
                                break
            
            # If no text from result, check captured stdout
            # The model often prints the OCR result to stdout
            if not extracted_text and captured_text:
                # Extract text from captured output
                # Look for the actual OCR text (usually after "=====================" markers)
                lines = captured_text.split('\n')
                # Find the text content (skip debug/status lines)
                text_lines = []
                skip_patterns = ['BASE:', 'PATCHES:', '=====================']
                for line in lines:
                    line = line.strip()
                    if line and not any(line.startswith(p) for p in skip_patterns):
                        text_lines.append(line)
                if text_lines:
                    extracted_text = '\n'.join(text_lines).strip()
            
            # If still no text, try converting result to string
            if not extracted_text and result is not None:
                result_str = str(result)
                if result_str and result_str != "None" and len(result_str) > 10:
                    extracted_text = result_str.strip()
            
            return extracted_text if extracted_text else ""

        except Exception as e:
            logger.warning(f"DeepSeek-OCR extraction failed: {e}. Falling back to Tesseract.")
            import traceback
            traceback.print_exc()
            return TesseractBackend().extract_text(image)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass


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
