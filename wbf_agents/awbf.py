"""Paper-aligned AWBF utilities.

All boxes in this module use xyxy=[x1, y1, x2, y2] unless a function name
explicitly says xywh. COCO export converts to xywh.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
import json
import math
import os
import warnings



@dataclass(frozen=True)
class Detection:
    box: Tuple[float, float, float, float]
    score: float
    label: int
    source: str = "model"
    reliability: float = 1.0


def _area_xyxy(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    inter = intersection_area(box_a, box_b)
    union = _area_xyxy(box_a) + _area_xyxy(box_b) - inter
    return 0.0 if union <= 0.0 else inter / union


def iob(subject_box: Sequence[float], reference_box: Sequence[float]) -> float:
    """Intersection over subject box area: IoB(subject|reference)."""
    denom = _area_xyxy(subject_box)
    return 0.0 if denom <= 0.0 else intersection_area(subject_box, reference_box) / denom


@dataclass
class IncrementalFusionState:
    label: int
    source: str = "incremental_awbf"
    total_weight: float = 0.0
    score_sum: float = 0.0
    count: int = 0
    fused_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    reliability: float = 1.0

    def add(self, detection: Detection) -> None:
        weight = max(0.0, detection.score)
        if self.count == 0:
            self.label = detection.label
            self.fused_box = tuple(float(v) for v in detection.box)
            self.total_weight = weight
        elif self.total_weight + weight == 0.0:
            self.fused_box = tuple(
                (self.fused_box[i] * self.count + detection.box[i]) / (self.count + 1)
                for i in range(4)
            )
        else:
            new_total = self.total_weight + weight
            self.fused_box = tuple(
                (self.fused_box[i] * self.total_weight + detection.box[i] * weight) / new_total
                for i in range(4)
            )
            self.total_weight = new_total
        self.score_sum += detection.score
        self.count += 1
        self.reliability = max(self.reliability, detection.reliability)

    def to_detection(self) -> Detection:
        if self.count == 0:
            raise ValueError("Cannot create detection from empty incremental fusion state")
        return Detection(
            tuple(float(v) for v in self.fused_box),
            self.score_sum / self.count,
            self.label,
            self.source,
            self.reliability,
        )


def incremental_fuse_detections(detections: Sequence[Detection], source: str = "incremental_awbf") -> Detection:
    if not detections:
        raise ValueError("detections must not be empty")
    state = IncrementalFusionState(label=detections[0].label, source=source)
    for detection in detections:
        state.add(detection)
    return state.to_detection()


def incremental_awbf(clusters: Sequence[Sequence[Detection]]) -> List[Detection]:
    return [incremental_fuse_detections(cluster) for cluster in clusters if cluster]

def cluster_detections_by_label_and_iou(
    detections: Sequence[Detection],
    iou_thr: float = 0.55,
    use_representative: bool = True,
) -> List[List[Detection]]:
    """Cluster detections by label and overlap before fusion/competition.

    Boxes never cluster across labels. A box joins the first same-label cluster if
    its IoU with the cluster representative (confidence-weighted box) or any
    member is at least ``iou_thr``.
    """
    clusters: List[List[Detection]] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        matched_idx = None
        for idx, cluster in enumerate(clusters):
            if cluster[0].label != det.label:
                continue
            candidates = [confidence_weighted_box(cluster)] if use_representative else cluster
            if any(iou(det.box, candidate.box) >= iou_thr for candidate in candidates):
                matched_idx = idx
                break
            if use_representative and any(iou(det.box, member.box) >= iou_thr for member in cluster):
                matched_idx = idx
                break
        if matched_idx is None:
            clusters.append([det])
        else:
            clusters[matched_idx].append(det)
    return clusters


def cluster_stats(detections: Sequence[Detection], clusters: Sequence[Sequence[Detection]]) -> Dict[str, float]:
    box_count = len(detections)
    cluster_count = len(clusters)
    largest = max((len(c) for c in clusters), default=0)
    avg = (box_count / cluster_count) if cluster_count else 0.0
    global_pairs = box_count * (box_count - 1) // 2
    clustered_pairs = sum(len(c) * (len(c) - 1) // 2 for c in clusters)
    return {
        "boxes_before_clustering": box_count,
        "clusters": cluster_count,
        "largest_cluster_size": largest,
        "average_cluster_size": avg,
        "global_pair_comparisons": global_pairs,
        "clustered_pair_comparisons": clustered_pairs,
        "comparisons_avoided_estimate": global_pairs - clustered_pairs,
    }

def confidence_weighted_box(detections: Sequence[Detection]) -> Detection:
    if not detections:
        raise ValueError("detections must not be empty")
    weights = [max(0.0, d.score) for d in detections]
    total = sum(weights)
    if total == 0.0:
        weights = [1.0 for _ in detections]
        total = float(len(detections))
    fused = tuple(sum(d.box[i] * w for d, w in zip(detections, weights)) / total for i in range(4))
    return Detection(tuple(float(x) for x in fused), sum(d.score for d in detections) / len(detections), detections[0].label, "fused", max(d.reliability for d in detections))


def decentralized_wbf(detections_by_model: Mapping[str, Sequence[Detection]], iou_thr: float = 0.55, skip_box_thr: float = 0.0) -> List[Detection]:
    """Run classical WBF through model-keyed detection lists, preserving label separation."""
    candidates = [d for detections in detections_by_model.values() for d in detections if d.score >= skip_box_thr]
    candidates.sort(key=lambda d: d.score, reverse=True)
    clusters: List[List[Detection]] = []
    for det in candidates:
        best_idx = None
        best_iou = iou_thr
        for idx, cluster in enumerate(clusters):
            fused = confidence_weighted_box(cluster)
            if det.label == fused.label:
                overlap = iou(det.box, fused.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
        if best_idx is None:
            clusters.append([det])
        else:
            clusters[best_idx].append(det)
    fused = [replace(confidence_weighted_box(cluster), source="awbf_wbf") for cluster in clusters]
    return sorted(fused, key=lambda d: d.score, reverse=True)


def compete_pair(attacker: Detection, defender: Detection, threshold: float = 0.5) -> List[Detection]:
    """Apply paper attack/defense rule to two detections."""
    if attacker.label != defender.label or iou(attacker.box, defender.box) <= 0.0:
        return [attacker, defender]
    attack = attacker.score * iob(defender.box, attacker.box)
    defense = defender.score * iob(attacker.box, defender.box)
    margin = attack - defense
    if margin > threshold:
        return [attacker]
    if margin < -threshold:
        return [defender]
    return [confidence_weighted_box([attacker, defender])]


def awbf_competition(detections: Sequence[Detection], iou_thr: float = 0.55, threshold: float = 0.5) -> List[Detection]:
    remaining = list(detections)
    changed = True
    while changed:
        changed = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if remaining[i].label == remaining[j].label and iou(remaining[i].box, remaining[j].box) >= iou_thr:
                    result = compete_pair(remaining[i], remaining[j], threshold)
                    remaining = [d for k, d in enumerate(remaining) if k not in (i, j)] + result
                    changed = True
                    break
            if changed:
                break
    return remaining


def auction_bid(a: Detection, b: Detection, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> float:
    utility = alpha * a.score + beta * iou(a.box, b.box) + gamma * a.reliability
    return utility * iob(b.box, a.box)


def negotiate_pair(a: Detection, b: Detection, rmax: int = 5, w: float = 0.3, threshold: float = 0.05) -> Tuple[Detection, int]:
    """Multi-round proposal adjustment. Returns final/fused detection and updates used."""
    if a.label != b.label:
        return a, 0
    current = a
    last_bid = -math.inf
    updates = 0
    for _ in range(max(1, rmax)):
        bid_current = auction_bid(current, b)
        bid_other = auction_bid(b, current)
        if abs(bid_current - bid_other) <= threshold:
            return confidence_weighted_box([current, b]), updates
        target = b if bid_other > bid_current else current
        if target is current or bid_current <= last_bid + 1e-12:
            break
        new_box = tuple((1.0 - w) * current.box[i] + w * target.box[i] for i in range(4))
        new_score = min(1.0, current.score + w * max(0.0, target.score - current.score))
        current = replace(current, box=new_box, score=new_score)
        last_bid = bid_current
        updates += 1
    return current, updates


def awbf_negotiation(
    detections: Sequence[Detection],
    iou_thr: float = 0.55,
    rmax: int = 5,
    w: float = 0.3,
    threshold: float = 0.05,
    progress_callback: Callable[[Dict[str, Any]], None] | None = None,
) -> List[Detection]:
    remaining = list(detections)
    changed = True
    pass_num = 0
    total_comparisons = 0
    while changed:
        pass_num += 1
        changed = False
        pass_comparisons = 0
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                pass_comparisons += 1
                total_comparisons += 1
                if remaining[i].label == remaining[j].label and iou(remaining[i].box, remaining[j].box) >= iou_thr:
                    fused, _updates = negotiate_pair(remaining[i], remaining[j], rmax=rmax, w=w, threshold=threshold)
                    remaining = [d for k, d in enumerate(remaining) if k not in (i, j)] + [fused]
                    changed = True
                    break
            if changed:
                break
        if progress_callback is not None:
            progress_callback({
                "pass": pass_num,
                "remaining_boxes": len(remaining),
                "pass_comparisons": pass_comparisons,
                "total_comparisons": total_comparisons,
                "changed": changed,
                "rmax": rmax,
            })
    return remaining


def load_coco_image_sizes(annotations_json: str) -> Dict[int, Tuple[float, float]]:
    with open(annotations_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(img["id"]): (float(img["width"]), float(img["height"])) for img in data.get("images", [])}


def bbox_xywh_looks_normalized(bbox: Sequence[float]) -> bool:
    x, y, w, h = [float(v) for v in bbox]
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0


def convert_coco_detection_bboxes(
    rows: Sequence[Dict[str, Any]],
    image_sizes: Mapping[int, Tuple[float, float]] | None = None,
    bbox_scale: str = "pixel",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if bbox_scale not in {"auto", "normalized", "pixel"}:
        raise ValueError("bbox_scale must be one of: auto, normalized, pixel")
    converted_rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "bbox_scale_mode": bbox_scale,
        "boxes_converted_to_pixel": 0,
        "boxes_left_unchanged": 0,
        "missing_image_ids": [],
        "small_width_height_boxes": 0,
    }
    missing_ids = set()
    normalized_in_pixel_mode = 0

    for row in rows:
        out = dict(row)
        bbox = [float(v) for v in out["bbox"]]
        image_id = int(out["image_id"])
        looks_normalized = bbox_xywh_looks_normalized(bbox)
        should_convert = bbox_scale == "normalized" or (bbox_scale == "auto" and looks_normalized)

        if bbox_scale == "pixel" and looks_normalized:
            normalized_in_pixel_mode += 1

        if should_convert:
            if not image_sizes or image_id not in image_sizes:
                missing_ids.add(image_id)
                stats["boxes_left_unchanged"] += 1
            else:
                width, height = image_sizes[image_id]
                out["bbox"] = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
                stats["boxes_converted_to_pixel"] += 1
        else:
            stats["boxes_left_unchanged"] += 1

        if out["bbox"][2] <= 1.0 or out["bbox"][3] <= 1.0:
            stats["small_width_height_boxes"] += 1
        converted_rows.append(out)

    stats["missing_image_ids"] = sorted(missing_ids)
    if missing_ids:
        warnings.warn(f"Missing image_id(s) in annotations for bbox scaling: {sorted(missing_ids)}")
    if normalized_in_pixel_mode:
        warnings.warn("bbox values look normalized while --bbox-scale pixel is selected")
    if converted_rows and stats["small_width_height_boxes"] / len(converted_rows) >= 0.5:
        warnings.warn("Many exported/evaluated bboxes have width/height <= 1.0; check --bbox-scale")
    return converted_rows, stats

def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    return [float(box[0]), float(box[1]), max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1]))]


def export_coco_detections(
    detections_by_image: Mapping[int, Sequence[Detection]],
    output_json: str,
    image_sizes: Mapping[int, Tuple[float, float]] | None = None,
    bbox_scale: str = "pixel",
    return_stats: bool = False,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for image_id, detections in detections_by_image.items():
        for d in detections:
            rows.append({"image_id": int(image_id), "category_id": int(d.label), "bbox": xyxy_to_xywh(d.box), "score": float(min(1.0, max(0.0, d.score)))})
    rows, stats = convert_coco_detection_bboxes(rows, image_sizes=image_sizes, bbox_scale=bbox_scale)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return (rows, stats) if return_stats else rows


def evaluate_coco(annotations_json: str, detections_json: str) -> Dict[str, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(annotations_json)
    coco_dt = coco_gt.loadRes(detections_json)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR", "AR50", "AR75", "AR_small", "AR_medium", "AR_large"]
    return dict(zip(keys, map(float, ev.stats)))
