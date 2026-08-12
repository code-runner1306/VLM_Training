import json
import logging
import time
from typing import Any, Dict, Optional
import torch
from PIL import Image

try:
    from vlm_annotation.src.models.base import ModelResponse, VisionModel, extract_json_from_text
except ImportError:
    from .base import ModelResponse, VisionModel, extract_json_from_text

try:
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None
    try:
        from transformers import AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        Qwen3VLForConditionalGeneration = None
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger("HuggingFaceVisionModel")


class HuggingFaceVisionModel(VisionModel):
    """
    Provider implementation for local Hugging Face Vision-Language Models.
    Supports Qwen series (Qwen3-VL-8B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen2.5-VL-3B-Instruct)
    and InternVL series (InternVL2.5-8B / InternVL3.5-8B, InternVL2.5-14B / InternVL3.5-14B).
    Loads base models directly from Hugging Face Hub using PyTorch & bitsandbytes 4-bit quantization.
    """

    def __init__(self, provider_name: str, model_id: str, config: Dict[str, Any]):
        super().__init__(provider_name, model_id, config)
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers and torch are required for HuggingFaceVisionModel.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        q_cfg = config.get("quantization", {})
        use_4bit = q_cfg.get("enabled", True) and torch.cuda.is_available()

        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "trust_remote_code": True,
        }

        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
            )

        # Load Processor / Tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        except Exception:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # Load Model Class with robust fallback for Vision-Language Models
        self.model = None

        if Qwen3VLForConditionalGeneration is not None and "qwen3" in self.model_id.lower():
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e:
                logger.warning(f"Qwen3VLForConditionalGeneration failed for {self.model_id}: {e}")

        if self.model is None and Qwen2_5_VLForConditionalGeneration is not None and ("qwen2.5" in self.model_id.lower() or "qwen2_5" in self.model_id.lower()):
            try:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e:
                logger.warning(f"Qwen2_5_VLForConditionalGeneration failed for {self.model_id}: {e}")

        if self.model is None and AutoModelForImageTextToText is not None:
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e:
                logger.warning(f"AutoModelForImageTextToText failed for {self.model_id}: {e}")

        if self.model is None and AutoModelForVision2Seq is not None:
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e:
                logger.warning(f"AutoModelForVision2Seq failed for {self.model_id}: {e}")

        if self.model is None and "internvl" in self.model_id.lower():
            try:
                self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            except Exception as e:
                logger.warning(f"AutoModel failed for {self.model_id}: {e}")

        if self.model is None:
            try:
                self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)

    async def generate_annotation(
        self,
        image_path: str,
        disease_name: str,
        prompt: str,
        disease_profile: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        self.total_requests += 1
        start_time = time.monotonic()

        try:
            pil_img = Image.open(image_path).convert("RGB")

            # Handling InternVL models vs Qwen models
            if "internvl" in self.model_id.lower() and hasattr(self.model, "chat"):
                # Native InternVL chat interface
                generation_config = dict(max_new_tokens=1000, do_sample=False)
                with torch.no_grad():
                    response_text, _ = self.model.chat(
                        self.tokenizer,
                        pil_img,
                        prompt,
                        generation_config
                    )
                raw_text = response_text
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_img},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                if self.processor and hasattr(self.processor, "apply_chat_template"):
                    formatted_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    formatted_prompt = prompt

                if self.processor:
                    inputs = self.processor(
                        text=[formatted_prompt],
                        images=[pil_img],
                        return_tensors="pt",
                        padding=True,
                    )
                else:
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

                inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=1000)

                if "input_ids" in inputs:
                    input_len = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, input_len:]

                if self.processor:
                    raw_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                else:
                    raw_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            latency_ms = (time.monotonic() - start_time) * 1000.0
            parsed_json = extract_json_from_text(raw_text)

            if parsed_json:
                self.successful_requests += 1
                self.total_latency_ms += latency_ms
                return ModelResponse(
                    provider=self.provider_name,
                    model_name=self.model_id,
                    raw_response=raw_text,
                    parsed_json=parsed_json,
                    latency_ms=latency_ms,
                    status="success"
                )
            else:
                self.json_parse_failures += 1
                self.failed_requests += 1
                return ModelResponse(
                    provider=self.provider_name,
                    model_name=self.model_id,
                    raw_response=raw_text,
                    parsed_json=None,
                    latency_ms=latency_ms,
                    status="json_parse_error",
                    error_message="Failed to parse valid JSON from HuggingFace VLM response"
                )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000.0
            self.failed_requests += 1
            err_msg = f"HuggingFace VLM generation error: {str(e)}"
            logger.error(err_msg)
            return ModelResponse(
                provider=self.provider_name,
                model_name=self.model_id,
                raw_response="",
                latency_ms=latency_ms,
                status="error",
                error_message=err_msg
            )
