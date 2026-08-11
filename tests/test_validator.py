from vlm_annotation.src.annotation.validator import AnnotationValidator


def test_validator_success():
    validator = AnnotationValidator("vlm_annotation/config/annotation_schema.json")
    valid_data = {
        "image_id": "12345678",
        "image_path": "Disease_A/img01.jpg",
        "disease": "Disease_A",
        "visible_observations": ["Yellow spots on lower leaf surface."],
        "affected_regions": ["leaves"],
        "color_characteristics": ["yellow", "brown"],
        "shape_characteristics": ["circular"],
        "texture_characteristics": ["dry"],
        "spatial_distribution": "scattered",
        "severity": "mild",
        "diagnostic_evidence": ["Concentric circular yellowing pattern"],
        "reasoning": "Observed circular yellow spots match early symptoms.",
        "uncertain_observations": [],
        "confidence": 0.9
    }

    is_valid, quality_status, msg = validator.validate(valid_data, "Disease_A")
    assert is_valid is True
    assert quality_status == "accepted"


def test_validator_disease_mismatch():
    validator = AnnotationValidator("vlm_annotation/config/annotation_schema.json")
    data = {
        "image_id": "12345678",
        "image_path": "Disease_A/img01.jpg",
        "disease": "Disease_B",  # Mismatch
        "visible_observations": ["Yellow spots on lower leaf surface."],
        "affected_regions": ["leaves"],
        "color_characteristics": ["yellow"],
        "shape_characteristics": ["circular"],
        "texture_characteristics": ["dry"],
        "spatial_distribution": "scattered",
        "severity": "mild",
        "diagnostic_evidence": ["Concentric circular pattern"],
        "reasoning": "Reasoning sentence long enough.",
        "uncertain_observations": [],
        "confidence": 0.9
    }

    is_valid, quality_status, msg = validator.validate(data, "Disease_A")
    assert is_valid is False
    assert quality_status == "failed"
    assert "mismatch" in msg.lower()


def test_validator_hallucination_warning():
    validator = AnnotationValidator("vlm_annotation/config/annotation_schema.json")
    data = {
        "image_id": "12345678",
        "image_path": "Disease_A/img01.jpg",
        "disease": "Disease_A",
        "visible_observations": ["Leaf shows spots. Recommend fungicide spray."],
        "affected_regions": ["leaves"],
        "color_characteristics": ["yellow"],
        "shape_characteristics": ["circular"],
        "texture_characteristics": ["dry"],
        "spatial_distribution": "scattered",
        "severity": "mild",
        "diagnostic_evidence": ["Concentric circular pattern"],
        "reasoning": "Reasoning sentence long enough.",
        "uncertain_observations": [],
        "confidence": 0.9
    }

    is_valid, quality_status, msg = validator.validate(data, "Disease_A")
    assert is_valid is True
    assert quality_status == "needs_review"
    assert "fungicide" in msg
