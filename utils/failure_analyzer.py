"""Failure mode analysis for translated SAR output (Phase 7, Step 7.3).

Given a set of quality metrics for a generated/translated image, attribute its
failure to a single dominant cause using simple, interpretable rules. The rules
mirror the guide:

    overlapping objects        -> "possible occlusion"
    high speckle variance      -> "high speckle noise"
    low contrast               -> "low contrast"
    otherwise                  -> "uncertain — no single dominant cause identified"

Speckle is the dominant noise in SAR; the variance of the (local) intensity
distribution estimates it. Low contrast means a nearly-uniform image, which a
GAN can produce when it collapses to the mean. Overlapping detected boxes hint
at occluded ships, where a detector (and the accompanying uncertainty) may
misread the scene.
"""

from __future__ import annotations


def _boxes_overlap(boxes, iou_thresh: float = 0.5) -> int:
    """Count pairs of boxes (xmin,ymin,xmax,ymax) with IoU above ``iou_thresh``."""
    n = len(boxes)
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
            ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = ix1 - ix0, iy1 - iy0
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union > iou_thresh:
                pairs += 1
    return pairs


def analyze_failure_mode(
    speckle_variance: float | None = None,
    contrast: float | None = None,
    boxes=None,
    speckle_thresh: float = 0.05,
    contrast_thresh: float = 0.1,
    iou_thresh: float = 0.5,
) -> str:
    """Attribute a failure to its dominant cause.

    Parameters
    ----------
    speckle_variance : float | None
        Variance of the normalized image intensity (SAR speckle proxy). High
        means noisy / speckled. `None` = metric not measured -> rule skipped.
    contrast : float | None
        (max - min) of the normalized intensity. Low means washed-out.
        `None` = rule skipped.
    boxes : list of (xmin, ymin, xmax, ymax) | None
        Detector boxes on the image. Overlapping pairs => occlusion.
        `None` = overlap not checked (e.g. no detections).
    speckle_thresh, contrast_thresh, iou_thresh
        Thresholds defining "high" speckle / "low" contrast / overlapping.

    Returns
    -------
    str
        One of the four explanations, most specific rule first.
    """
    # occlusion is the most specific (depends on actual detections); check first.
    if boxes is not None and _boxes_overlap(boxes, iou_thresh) > 0:
        return "possible occlusion"

    if speckle_variance is not None and speckle_variance > speckle_thresh:
        return "high speckle noise"

    if contrast is not None and contrast < contrast_thresh:
        return "low contrast"

    return "uncertain — no single dominant cause identified"


if __name__ == "__main__":
    # Synthetic cases: one per rule, each must return its distinct explanation.
    cases = {
        "possible occlusion": dict(boxes=[(10, 10, 60, 60), (20, 20, 60, 60)],
                                   speckle_variance=0.001, contrast=0.8),
        "high speckle noise": dict(speckle_variance=0.3, contrast=0.8, boxes=None),
        "low contrast": dict(speckle_variance=0.01, contrast=0.01, boxes=None),
        "uncertain — no single dominant cause identified": dict(
            speckle_variance=0.01, contrast=0.8, boxes=None),
    }

    ok = True
    for expected, kw in cases.items():
        got = analyze_failure_mode(**kw)
        status = "PASS" if got == expected else "FAIL"
        ok &= got == expected
        print(f"  {status}: expected {expected!r}  ->  got {got!r}")
    print("OK: every synthetic case returns its correct, distinct explanation."
          if ok else "FAIL: at least one case returned the wrong label.")