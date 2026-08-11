import asyncio
import tempfile
from pathlib import Path
from PIL import Image
from vlm_annotation.src.evaluation.benchmark import sample_benchmark_images
from vlm_annotation.src.evaluation.scoring import BenchmarkEvaluator


def test_stratified_benchmark_sampling():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        d_a = root / "Disease_A"
        d_b = root / "Disease_B"
        d_a.mkdir()
        d_b.mkdir()

        for i in range(10):
            img = Image.new("RGB", (50, 50))
            img.save(d_a / f"img_a_{i}.jpg")
            img.save(d_b / f"img_b_{i}.jpg")
            img.close()

        samples = sample_benchmark_images(
            dataset_dir=tmpdir,
            target_count=6,
            seed=42,
            output_path=str(root / "samples.json")
        )

        assert len(samples) == 6
        diseases = [s.disease_name for s in samples]
        assert diseases.count("Disease_A") == 3
        assert diseases.count("Disease_B") == 3


def test_benchmark_heuristic_scoring():
    async def _test():
        evaluator = BenchmarkEvaluator()
        candidate = {
            "disease": "Disease_A",
            "visible_observations": [
                "Small yellow spots on upper leaf surface.",
                "Necrotic leaf margins.",
                "Concentrated chlorosis along main vein.",
                "Rough leaf surface texture."
            ],
            "diagnostic_evidence": [
                "Visual features correspond to early blight disease.",
                "Concentric leaf spot halos matched."
            ],
            "reasoning": "Observed spot dimensions and necrotic leaf margins are consistent with Disease_A symptoms over time."
        }

        res = await evaluator.evaluate_annotation(
            image_path="dummy.jpg",
            disease_name="Disease_A",
            candidate_json=candidate,
            raw_response=""
        )

        assert res.ground_truth_matched is True
        assert res.hallucination_detected is False
        assert res.total_score > 60.0

    asyncio.run(_test())
