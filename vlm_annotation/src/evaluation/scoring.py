import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from vlm_annotation.src.models.base import VisionModel

logger = logging.getLogger("vlm_annotation.scoring")


@dataclass
class ScoreResult:
    visual_observation_score: float  # max 30
    diagnostic_evidence_score: float  # max 25
    reasoning_score: float            # max 20
    hallucination_score: float        # max 15
    schema_score: float               # max 10
    total_score: float                # max 100
    hallucination_detected: bool
    ground_truth_matched: bool
    judge_feedback: str


class BenchmarkEvaluator:
    """Evaluates candidate annotations using rules and Teacher-as-Judge LLM/VLM calls."""

    def __init__(self, judge_model: Optional[VisionModel] = None):
        self.judge_model = judge_model
        prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "sugarcane_prompt" / "benchmark.txt"
        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.judge_prompt_template = f.read()
        else:
            self.judge_prompt_template = ""

    async def evaluate_annotation(
        self,
        image_path: str,
        disease_name: str,
        candidate_json: Optional[Dict[str, Any]],
        raw_response: str
    ) -> ScoreResult:
        # Rule-based fallback scoring if candidate_json is None or invalid
        if not candidate_json:
            return ScoreResult(
                visual_observation_score=0.0,
                diagnostic_evidence_score=0.0,
                reasoning_score=0.0,
                hallucination_score=0.0,
                schema_score=0.0,
                total_score=0.0,
                hallucination_detected=True,
                ground_truth_matched=False,
                judge_feedback="Failed JSON parsing or empty candidate response."
            )

        schema_score = 10.0
        gt_matched = candidate_json.get("disease", "").lower() == disease_name.lower()
        if not gt_matched:
            schema_score = 5.0

        obs = candidate_json.get("visible_observations", [])
        evidence = candidate_json.get("diagnostic_evidence", [])
        reasoning = candidate_json.get("reasoning", "")

        # Check basic heuristic bounds
        obs_score = min(30.0, len(obs) * 7.5) if obs else 0.0
        evid_score = min(25.0, len(evidence) * 8.33) if evidence else 0.0
        reas_score = min(20.0, len(reasoning.split()) * 0.5) if reasoning else 0.0

        # Simple heuristic hallucination check
        hallucinatory_words = ["fungicide", "microscope", "dna", "pesticide", "soil ph", "temperature", "humidity"]
        text_full = json.dumps(candidate_json).lower()
        hallucination_found = any(w in text_full for w in hallucinatory_words)
        halluc_score = 0.0 if hallucination_found else 15.0

        # Teacher-as-Judge evaluation if judge model available
        if self.judge_model and self.judge_prompt_template:
            try:
                judge_prompt = self.judge_prompt_template.replace(
                    "{DISEASE_NAME}", disease_name
                ).replace(
                    "{CANDIDATE_JSON}", json.dumps(candidate_json, indent=2)
                )

                judge_response = await self.judge_model.generate_annotation(
                    image_path=image_path,
                    disease_name=disease_name,
                    prompt=judge_prompt
                )

                if judge_response.status == "success" and judge_response.parsed_json:
                    pj = judge_response.parsed_json
                    return ScoreResult(
                        visual_observation_score=float(pj.get("visual_observation_score", obs_score)),
                        diagnostic_evidence_score=float(pj.get("diagnostic_evidence_score", evid_score)),
                        reasoning_score=float(pj.get("reasoning_score", reas_score)),
                        hallucination_score=float(pj.get("hallucination_score", halluc_score)),
                        schema_score=float(pj.get("schema_score", schema_score)),
                        total_score=float(pj.get("total_score", obs_score + evid_score + reas_score + halluc_score + schema_score)),
                        hallucination_detected=bool(pj.get("hallucination_detected", hallucination_found)),
                        ground_truth_matched=gt_matched,
                        judge_feedback=str(pj.get("judge_feedback", "Evaluated by Teacher-as-Judge."))
                    )
            except Exception as e:
                logger.warning(f"Judge model evaluation failed: {e}. Falling back to rule-based scoring.")

        total = round(obs_score + evid_score + reas_score + halluc_score + schema_score, 2)
        return ScoreResult(
            visual_observation_score=round(obs_score, 2),
            diagnostic_evidence_score=round(evid_score, 2),
            reasoning_score=round(reas_score, 2),
            hallucination_score=round(halluc_score, 2),
            schema_score=round(schema_score, 2),
            total_score=total,
            hallucination_detected=hallucination_found,
            ground_truth_matched=gt_matched,
            judge_feedback="Rule-based heuristic evaluation."
        )
