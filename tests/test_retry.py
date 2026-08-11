import asyncio
from vlm_annotation.src.annotation.retry import RateLimiter, execute_with_retry
from vlm_annotation.src.models.base import VisionModel


class DummyModel(VisionModel):
    def __init__(self):
        super().__init__("dummy", "dummy-model", {})
        self.call_count = 0

    async def generate_annotation(self, image_path, disease_name, prompt, disease_profile=None):
        self.call_count += 1
        if self.call_count == 1:
            raise Exception("HTTP 429 Rate Limit Exceeded")
        return "Success"


def test_retry_increments_429_counters():
    async def _test():
        limiter = RateLimiter(requests_per_minute=600, max_concurrency=2)
        dummy = DummyModel()

        res = await execute_with_retry(
            dummy.generate_annotation,
            image_path="test.jpg",
            disease_name="test",
            prompt="test",
            rate_limiter=limiter,
            max_retries=3,
            initial_backoff=0.01,
            model_instance=dummy
        )

        assert res == "Success"
        assert dummy.call_count == 2
        assert dummy.rate_limit_hits == 1

    asyncio.run(_test())
