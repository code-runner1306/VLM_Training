# Final VLM Model Recommendation Report

## Overview of Findings

- **Overall Best Candidate Model:** `qwen25vl-3b-v1` (Composite Score: `100.0`/100)
- **Best Classification Candidate:** `qwen25vl-3b-v1` (Macro F1: `1.0000`)
- **Best Explanation Quality Candidate:** `qwen25vl-3b-v1` (Visual Grounding: `1.0000`)
- **Best Efficiency / VRAM Trade-off:** `qwen25vl-3b-v1` (Peak VRAM: `0.00 GB`)

## Data-Driven Justification

The model `qwen25vl-3b-v1` demonstrated superior balanced performance across diagnostic accuracy, visual evidence grounding, and computational resource requirements. It achieved an accuracy of `1.0000` and a macro F1 of `1.0000` while maintaining a visual grounding score of `1.0000`.

## Comparative Ranking Table

| Rank | Experiment | Model | Strategy | Accuracy | Macro F1 | Grounding | Peak VRAM | Composite Score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| #1 | `qwen25vl-3b-v1` | `qwen25vl-3b-v1` | `llm_projector` | `1.0000` | `1.0000` | `1.0000` | `0.00GB` | **100.0** |
| #2 | `qwen25vl-7b-v1` | `qwen25vl-7b-v1` | `llm_projector` | `1.0000` | `1.0000` | `1.0000` | `0.00GB` | **100.0** |

## Strategic Recommendation

We recommend selecting `qwen25vl-3b-v1` for production deployment and downstream integration into the cotton disease visual assistance system.
