"""
Semantic Consistency Evaluation Engine for ModelShield using RAM++.

Evaluates semantic tag retention, tag loss, tag injection, and Jaccard similarity
between clean and adversarial images using the Recognize Anything Plus Model (RAM++).
Completely decoupled from UI code for testability and headless execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from PIL import Image
import torch


@dataclass
class SemanticConsistencyResult:
    """Detailed result of semantic tag comparison between clean and adversarial inputs."""
    clean_tags: List[str] = field(default_factory=list)
    adversarial_tags: List[str] = field(default_factory=list)
    overlap_tags: List[str] = field(default_factory=list)
    lost_tags: List[str] = field(default_factory=list)
    new_tags: List[str] = field(default_factory=list)
    overlap_ratio: float = 1.0          # Jaccard similarity in [0.0, 1.0]
    tags_lost_count: int = 0
    new_tags_count: int = 0
    semantic_consistency: float = 100.0  # Percentage in [0.0, 100.0]
    clean_tags_count: int = 0
    adversarial_tags_count: int = 0
    tag_retention_rate: float = 100.0   # Percentage of clean tags preserved in [0.0, 100.0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_tags": list(self.clean_tags),
            "adversarial_tags": list(self.adversarial_tags),
            "overlap_tags": list(self.overlap_tags),
            "lost_tags": list(self.lost_tags),
            "new_tags": list(self.new_tags),
            "overlap_ratio": round(float(self.overlap_ratio), 4),
            "tags_lost_count": int(self.tags_lost_count),
            "new_tags_count": int(self.new_tags_count),
            "semantic_consistency": round(float(self.semantic_consistency), 2),
            "clean_tags_count": int(self.clean_tags_count),
            "adversarial_tags_count": int(self.adversarial_tags_count),
            "tag_retention_rate": round(float(self.tag_retention_rate), 2),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> SemanticConsistencyResult:
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            clean_tags=list(data.get("clean_tags", [])),
            adversarial_tags=list(data.get("adversarial_tags", [])),
            overlap_tags=list(data.get("overlap_tags", [])),
            lost_tags=list(data.get("lost_tags", [])),
            new_tags=list(data.get("new_tags", [])),
            overlap_ratio=float(data.get("overlap_ratio", 1.0)),
            tags_lost_count=int(data.get("tags_lost_count", 0)),
            new_tags_count=int(data.get("new_tags_count", 0)),
            semantic_consistency=float(data.get("semantic_consistency", 100.0)),
            clean_tags_count=int(data.get("clean_tags_count", len(data.get("clean_tags", [])))),
            adversarial_tags_count=int(data.get("adversarial_tags_count", len(data.get("adversarial_tags", [])))),
            tag_retention_rate=float(data.get("tag_retention_rate", 100.0)),
        )


def _normalize_tags(raw_tags: Union[List[str], Set[str], str, None]) -> List[str]:
    """
    Normalizes a tag collection into a sorted, unique list of lower-case strings.
    Handles delimiters (',', '|', '/'), mixed casing, leading/trailing whitespace,
    and filters out empty strings.
    """
    if raw_tags is None:
        return []

    cleaned: Set[str] = set()

    if isinstance(raw_tags, str):
        # Normalize all delimiters to '|' then split
        norm_str = raw_tags.replace(",", "|").replace("/", "|")
        for token in norm_str.split("|"):
            t = token.strip().lower()
            if t:
                cleaned.add(t)
    elif isinstance(raw_tags, (list, tuple, set)):
        for item in raw_tags:
            if item is None:
                continue
            if isinstance(item, str):
                norm_str = item.replace(",", "|").replace("/", "|")
                for token in norm_str.split("|"):
                    t = token.strip().lower()
                    if t:
                        cleaned.add(t)
            else:
                t = str(item).strip().lower()
                if t:
                    cleaned.add(t)
    return sorted(list(cleaned))


def compare_tag_sets(
    clean_tags: Union[List[str], Set[str], str, None],
    adversarial_tags: Union[List[str], Set[str], str, None],
) -> SemanticConsistencyResult:
    """
    Compares clean and adversarial tag sets (or baseline and candidate tag sets).

    Calculates:
      - Overlap tags: Clean ∩ Adversarial
      - Lost tags: Clean \\ Adversarial
      - New tags: Adversarial \\ Clean
      - Overlap ratio (Jaccard similarity): |C ∩ A| / |C ∪ A| (1.0 if both empty)
      - Semantic consistency percentage: Jaccard * 100.0
      - Tag retention rate: (|C ∩ A| / |C|) * 100.0 if |C| > 0 else (100.0 if |A| == 0 else 0.0)
    """
    c_list = _normalize_tags(clean_tags)
    a_list = _normalize_tags(adversarial_tags)

    c_set = set(c_list)
    a_set = set(a_list)

    overlap_set = c_set.intersection(a_set)
    lost_set = c_set.difference(a_set)
    new_set = a_set.difference(c_set)
    union_set = c_set.union(a_set)

    if len(union_set) == 0:
        # Both clean and adversarial tag sets are completely empty
        jaccard = 1.0
        consistency_pct = 100.0
        retention_pct = 100.0
    else:
        jaccard = len(overlap_set) / float(len(union_set))
        consistency_pct = jaccard * 100.0
        if len(c_set) > 0:
            retention_pct = (len(overlap_set) / float(len(c_set))) * 100.0
        else:
            retention_pct = 0.0

    return SemanticConsistencyResult(
        clean_tags=c_list,
        adversarial_tags=a_list,
        overlap_tags=sorted(list(overlap_set)),
        lost_tags=sorted(list(lost_set)),
        new_tags=sorted(list(new_set)),
        overlap_ratio=round(jaccard, 4),
        tags_lost_count=len(lost_set),
        new_tags_count=len(new_set),
        semantic_consistency=round(consistency_pct, 2),
        clean_tags_count=len(c_set),
        adversarial_tags_count=len(a_set),
        tag_retention_rate=round(retention_pct, 2),
    )


def evaluate_semantic_consistency_images(
    clean_image: Union[Image.Image, torch.Tensor],
    adversarial_image: Union[Image.Image, torch.Tensor],
    model: torch.nn.Module,
    preprocess: Any,
    device: Optional[torch.device] = None,
) -> SemanticConsistencyResult:
    """
    Runs RAM++ on clean and adversarial images to extract tags and compute semantic consistency.
    """
    target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(img, Image.Image):
            rgb = img.convert("RGB")
            t = preprocess(rgb)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t.to(target_device)
        elif isinstance(img, torch.Tensor):
            t = img.detach()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            # If shape is [1, 3, 32, 32] or other, interpolate/resize to 384x384 for RAM++
            if t.shape[-2:] != (384, 384):
                t = torch.nn.functional.interpolate(
                    t, size=(384, 384), mode="bicubic", align_corners=False
                )
            return t.to(target_device)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    clean_tensor = _to_tensor(clean_image)
    adv_tensor = _to_tensor(adversarial_image)

    from ram import inference

    with torch.no_grad():
        clean_en, _ = inference.inference_ram(clean_tensor, model)
        adv_en, _ = inference.inference_ram(adv_tensor, model)

    return compare_tag_sets(clean_en, adv_en)


def evaluate_semantic_regression(
    candidate_result: Optional[Union[SemanticConsistencyResult, Dict[str, Any]]],
    baseline_result: Optional[Union[SemanticConsistencyResult, Dict[str, Any]]] = None,
    min_semantic_consistency: float = 60.0,
    warn_semantic_consistency: float = 75.0,
    max_semantic_consistency_drop: float = 10.0,
    warn_semantic_consistency_drop: float = 5.0,
    enabled: bool = True,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
    """
    Evaluates semantic consistency regression between candidate and baseline.

    Checks:
      1. Candidate semantic consistency against absolute thresholds (min & warn).
      2. Candidate degradation compared to baseline (if baseline is present).

    Returns:
      (failure_message, warning_message, assessment_dict)
    """
    if not enabled:
        return None, None, None

    if candidate_result is None:
        # If enabled but candidate has no semantic consistency data
        warn_msg = "Candidate snapshot does not contain semantic consistency evaluation data."
        assessment = {
            "status": "MISSING",
            "summary": warn_msg,
            "candidate_consistency": None,
            "baseline_consistency": None,
        }
        return None, warn_msg, assessment

    cand_obj = (
        candidate_result
        if isinstance(candidate_result, SemanticConsistencyResult)
        else SemanticConsistencyResult.from_dict(candidate_result)
    )

    base_obj = None
    if baseline_result is not None:
        base_obj = (
            baseline_result
            if isinstance(baseline_result, SemanticConsistencyResult)
            else SemanticConsistencyResult.from_dict(baseline_result)
        )

    cand_score = cand_obj.semantic_consistency
    base_score = base_obj.semantic_consistency if base_obj is not None else None

    assessment = {
        "candidate_consistency": cand_score,
        "baseline_consistency": base_score,
        "min_threshold": min_semantic_consistency,
        "candidate_details": cand_obj.to_dict(),
        "baseline_details": base_obj.to_dict() if base_obj else None,
    }

    # 1. Absolute threshold checks
    if cand_score < min_semantic_consistency:
        fail_msg = (
            f"Candidate semantic consistency ({cand_score:.2f}%) fell below minimum "
            f"allowable threshold of {min_semantic_consistency:.2f}%."
        )
        assessment["status"] = "FAIL"
        assessment["summary"] = fail_msg
        return fail_msg, None, assessment

    if cand_score < warn_semantic_consistency:
        warn_msg = (
            f"Candidate semantic consistency ({cand_score:.2f}%) fell below warning "
            f"threshold of {warn_semantic_consistency:.2f}%."
        )
        assessment["status"] = "WARN"
        assessment["summary"] = warn_msg
        return None, warn_msg, assessment

    # 2. Comparative degradation checks vs baseline
    if base_score is not None:
        drop = base_score - cand_score
        assessment["consistency_drop"] = round(drop, 2)
        if drop > max_semantic_consistency_drop:
            fail_msg = (
                f"Candidate semantic consistency degraded by {drop:.2f}% (from {base_score:.2f}% "
                f"to {cand_score:.2f}%), exceeding max allowable drop of {max_semantic_consistency_drop:.2f}%."
            )
            assessment["status"] = "FAIL"
            assessment["summary"] = fail_msg
            return fail_msg, None, assessment
        elif drop > warn_semantic_consistency_drop:
            warn_msg = (
                f"Candidate semantic consistency degraded by {drop:.2f}% (from {base_score:.2f}% "
                f"to {cand_score:.2f}%), exceeding warning drop threshold of {warn_semantic_consistency_drop:.2f}%."
            )
            assessment["status"] = "WARN"
            assessment["summary"] = warn_msg
            return None, warn_msg, assessment

    # Passed all criteria
    assessment["status"] = "PASS"
    assessment["summary"] = (
        f"Candidate meets semantic consistency requirements ({cand_score:.2f}% consistency, "
        f"{cand_obj.tags_lost_count} tags lost, {cand_obj.new_tags_count} new tags)."
    )
    return None, None, assessment

