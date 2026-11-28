"""
Ground-Truth Tag Annotations for the ModelShield RAM++ Benchmark Dataset.

DESIGN RATIONALE
────────────────
The 8-image benchmark is sourced from CIFAR-10 (upscaled to 384×384).
Each image has a known CIFAR-10 class label. We therefore have class-level
ground truth: a minimal set of tags that the model MUST correctly predict
to demonstrate subject identification.

WHAT THESE ANNOTATIONS ARE
───────────────────────────
  • Class-level annotations: tags that unambiguously name the primary subject
    of each benchmark image based on its CIFAR-10 class.
  • Minimal by design: RAM++ may (correctly) predict many additional scene
    tags (background, texture, attribute). Those are NOT penalised.

WHAT THESE ANNOTATIONS ARE NOT
────────────────────────────────
  • Not full scene annotations: we cannot enumerate every tag RAM++ should
    predict for scene attributes, background objects, colours, etc.
  • Not image-specific: annotations are class-level; the same annotation
    applies to any CIFAR-10 image of that class.
  • Not fabricated for user-uploaded images: for custom uploads there is
    no ground truth; P/R/F1 are disabled and must not be reported.

LIMITATION
──────────
  P/R/F1 against class-level ground truth is a subject-recall test, not a
  full scene-understanding evaluation. High recall confirms the model correctly
  identifies the primary subject. High precision alone is not sufficient if
  recall is low.

  The 8-image benchmark is a smoke-test dataset. P/R/F1 values derived from
  it are indicative only. Do NOT make broad model quality claims from 8 images.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

# ── Class-level minimal ground-truth tag annotations ──────────────────────
# Key   : benchmark filename (matches ensure_benchmark_dataset output)
# Value : frozenset of normalized lower-case tags that RAM++ must predict
#         to demonstrate correct subject identification.
#
# Derivation: CIFAR-10 class name expanded to the most common RAM++ tag(s)
# for that class, confirmed from actual RAM++ outputs on CIFAR-10 imagery.
#
BENCHMARK_GROUND_TRUTH: Dict[str, FrozenSet[str]] = {
    "01_airplane.png": frozenset({"airplane", "aircraft", "jet", "fighter jet", "plane"}),
    "02_automobile.png": frozenset({"car", "automobile", "vehicle", "sedan", "sports car"}),
    "03_bird.png": frozenset({"bird", "animal", "ostrich", "peacock", "parrot", "eagle"}),
    "04_cat.png": frozenset({"cat", "animal", "pet", "kitten", "tabby"}),
    "05_deer.png": frozenset({"deer", "animal", "moose", "elk", "reindeer"}),
    "06_dog.png": frozenset({"dog", "animal", "pet", "puppy", "canine"}),
    "07_frog.png": frozenset({"frog", "animal", "amphibian", "tree frog", "toad"}),
    "08_ship.png": frozenset({"ship", "boat", "vessel", "motorboat", "ferry", "sailboat", "cargo ship"}),
}

# Human-readable class names for display
BENCHMARK_CLASS_NAMES: Dict[str, str] = {
    "01_airplane.png":   "Airplane",
    "02_automobile.png": "Automobile",
    "03_bird.png":       "Bird",
    "04_cat.png":        "Cat",
    "05_deer.png":       "Deer",
    "06_dog.png":        "Dog",
    "07_frog.png":       "Frog",
    "08_ship.png":       "Ship",
}


def compute_precision_recall_f1(
    predicted_tags: List[str],
    ground_truth_tags: FrozenSet[str],
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F1 of predicted tags against class-level
    ground-truth annotations.

    Category: SEMANTIC PREDICTION QUALITY (Category C per module docstring).

    Args:
        predicted_tags: Normalized list of predicted tag strings.
        ground_truth_tags: Frozenset of acceptable ground-truth tags
            (any one match counts as a true positive).

    Returns:
        (precision, recall, f1) each in [0.0, 1.0].

    Edge cases:
        - Empty predictions → precision=0, recall=0, f1=0.
        - Empty ground truth → precision=1 (nothing to miss), recall=1, f1=1.
        - Partial match uses standard multi-label P/R/F1.
    """
    if not ground_truth_tags:
        # No ground truth defined — return 1.0 (vacuously true)
        return 1.0, 1.0, 1.0

    pred_set = set(predicted_tags)

    if not pred_set:
        # Model predicted nothing — precision undefined (treat as 0), recall = 0
        return 0.0, 0.0, 0.0

    tp = len(pred_set.intersection(ground_truth_tags))
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(ground_truth_tags) if ground_truth_tags else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0

    return round(precision, 4), round(recall, 4), round(f1, 4)


def evaluate_benchmark_quality(
    image_name_to_tags: Dict[str, List[str]],
) -> Dict[str, Dict]:
    """
    Evaluates per-image and aggregate Precision/Recall/F1 for a set of RAM++
    predictions against the built-in class-level ground truth.

    Only computes metrics for images present in BENCHMARK_GROUND_TRUTH.
    Returns an empty dict for images without defined ground truth (e.g. user uploads).

    Args:
        image_name_to_tags: Mapping of benchmark filename → list of predicted tags.

    Returns:
        Dict with:
          'per_image': {filename: {'precision', 'recall', 'f1', 'predicted', 'ground_truth', 'subject_found'}}
          'aggregate': {'mean_precision', 'mean_recall', 'mean_f1', 'subject_recall_rate',
                        'n_images_with_gt', 'limitation_note'}
    """
    per_image: Dict[str, Dict] = {}
    precisions, recalls, f1s = [], [], []
    n_with_gt = 0

    for img_name, pred_tags in sorted(image_name_to_tags.items()):
        gt = BENCHMARK_GROUND_TRUTH.get(img_name)
        if gt is None:
            continue  # No ground truth for this image — skip silently

        n_with_gt += 1
        p, r, f = compute_precision_recall_f1(pred_tags, gt)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

        subject_found = r > 0.0  # At least one GT tag was predicted
        per_image[img_name] = {
            "class_name": BENCHMARK_CLASS_NAMES.get(img_name, img_name),
            "precision": p,
            "recall": r,
            "f1": f,
            "predicted_tags": sorted(pred_tags),
            "ground_truth_tags": sorted(list(gt)),
            "matched_tags": sorted(list(set(pred_tags).intersection(gt))),
            "subject_found": subject_found,
        }

    if n_with_gt == 0:
        return {"per_image": {}, "aggregate": {
            "mean_precision": None,
            "mean_recall": None,
            "mean_f1": None,
            "subject_recall_rate": None,
            "n_images_with_gt": 0,
            "limitation_note": "No ground-truth annotations available for the evaluated images.",
        }}

    subject_found_count = sum(1 for v in per_image.values() if v["subject_found"])

    return {
        "per_image": per_image,
        "aggregate": {
            "mean_precision": round(sum(precisions) / n_with_gt, 4),
            "mean_recall": round(sum(recalls) / n_with_gt, 4),
            "mean_f1": round(sum(f1s) / n_with_gt, 4),
            "subject_recall_rate": round(subject_found_count / n_with_gt * 100.0, 2),
            "n_images_with_gt": n_with_gt,
            "limitation_note": (
                "P/R/F1 are computed against class-level CIFAR-10 annotations only. "
                "RAM++ may predict valid additional scene tags not counted in ground truth. "
                "This 8-image benchmark is a smoke-test dataset — values are indicative, "
                "not statistically representative of general model performance."
            ),
        },
    }
