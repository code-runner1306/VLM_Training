import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import vlm_annotation.src.models.nvidia_nim as nvidia_mod
from vlm_annotation.src.models.factory import create_vision_model
from vlm_annotation.src.models.nvidia_nim import NvidiaVisionModel


def test_mock_nvidia_provider():
    async def _test():
        cfg = {
            "provider": "nvidia",
            "model": "meta/llama-3.2-90b-vision-instruct",
            "name": "nvidia-llama-3.2-90b"
        }

        with patch.object(nvidia_mod, "HAS_OPENAI", True):
            with patch.dict("os.environ", {"NVIDIA_API_KEY": "mock-nvidia-key"}):
                with patch("vlm_annotation.src.models.nvidia_nim.AsyncOpenAI") as mock_openai:
                    mock_client = MagicMock()
                    mock_openai.return_value = mock_client

                    mock_choice = MagicMock()
                    mock_choice.message.content = '{"disease": "Disease_A", "visible_observations": ["spot"], "diagnostic_evidence": ["spot"], "reasoning": "valid reasoning text for disease"}'
                    mock_resp = MagicMock()
                    mock_resp.choices = [mock_choice]
                    mock_resp.usage.prompt_tokens = 100
                    mock_resp.usage.completion_tokens = 50

                    mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)

                    model = create_vision_model(cfg)
                    assert isinstance(model, NvidiaVisionModel)

                    with patch("vlm_annotation.src.models.nvidia_nim.validate_and_prepare_image", return_value=b"fakebytes"):
                        response = await model.generate_annotation(
                            image_path="fake.jpg",
                            disease_name="Disease_A",
                            prompt="Describe image"
                        )

                        assert response.status == "success"
                        assert response.parsed_json["disease"] == "Disease_A"
                        assert model.successful_requests == 1
                        assert model.rate_limit_hits == 0

    asyncio.run(_test())
