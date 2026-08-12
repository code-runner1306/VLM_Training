import json
from pathlib import Path
from typing import Any, Dict, Tuple
import jsonschema

PROHIBITED_HALLUCINATIONS = [
    "fungicide", "pesticide", "spray", "chemical treatment", "dna",
    "microscope", "microscopic", "bacterial strain", "soil ph",
    "soil moisture", "temperature range", "relative humidity", "pathogen spore count"
]


class AnnotationValidator:
    def __init__(self, schema_path: str = None):
        if schema_path is None:
            schema_path = "vlm_annotation/config/annotation_schema_sugarcane.json"
        sp = Path(schema_path)
        if not sp.exists():
            sp = Path(__file__).resolve().parent.parent.parent / "config" / "annotation_schema_sugarcane.json"
        with open(sp, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def validate(self, data: Dict[str, Any], ground_truth_disease: str) -> Tuple[bool, str, str]:
        """
        Validate annotation dict against schema, ground truth label, and hallucination rules.
        Returns: (is_valid, quality_status, error_or_reason)
        quality_status: 'accepted', 'needs_review', 'failed'
        """
        # 1. JSON Schema Check
        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as e:
            return False, "failed", f"Schema validation failed: {e.message}"
        except Exception as e:
            return False, "failed", f"Schema error: {str(e)}"

        # 2. Disease Label Match
        if data.get("disease", "").strip().lower() != ground_truth_disease.strip().lower():
            return False, "failed", f"Disease label mismatch: expected '{ground_truth_disease}', got '{data.get('disease')}'"

        # 3. Non-empty Observations Check
        obs = data.get("visible_observations", [])
        # Sugarcane schema uses visual_evidence; fall back to cotton diagnostic_evidence if present.
        evidence = data.get("visual_evidence") or data.get("diagnostic_evidence", [])
        if not obs or len(obs) == 0:
            return False, "failed", "Empty visible_observations list."
        if not evidence or len(evidence) == 0:
            return False, "failed", "Empty evidence list (visual_evidence)."

        # 4. Hallucination Inspection
        text_content = json.dumps(data).lower()
        found_hallucinations = [w for w in PROHIBITED_HALLUCINATIONS if w in text_content]
        if found_hallucinations:
            return True, "needs_review", f"Hallucination warning: found prohibited non-visual terms {found_hallucinations}"

        return True, "accepted", "Valid annotation grounded in visual evidence."