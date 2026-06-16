# PAPER_SPEC.md

## Purpose

This file is the authoritative paper-derived specification for auditing and fixing this repository.

Codex must verify whether the fork implements the paper:

**Adaptive Multi-Agent Approach for Bounding-Box Fusion in Road-Scene Object Detection**

Do not request or use the original PDF. Use this file as the implementation and reproducibility target.

---

## Core Goal

The repository should implement and reproduce a multi-agent bounding-box fusion framework called **AWBF**.

AWBF extends classical **Weighted Boxes Fusion (WBF)** by representing bounding boxes as agents that can:

1. Cooperate through decentralized WBF.
2. Compete through attack-defense / auction-like interactions.
3. Negotiate through multi-round bounding-box and confidence adjustment.

The target task is object-detection bounding-box fusion on COCO-style predictions.

---

## Required System Architecture

The implementation should contain the following conceptual agents.

### 1. Bounding-Box Agents

Each bounding-box agent represents one detection proposal.

Each agent must store:

- bounding-box coordinates
- confidence score
- object label/category
- source detector/model
- optional model reliability score
- current lifecycle state

Expected responsibilities:

- find overlapping boxes
- filter candidate boxes
- compare confidence and category consistency
- participate in fusion, competition, or negotiation
- post final result to blackboard/shared state

### 2. Model-Specific Agents

Each model-specific agent manages detections from one detector.

Expected responsibilities:

- normalize detector-specific output format
- provide bounding boxes, labels, and scores
- optionally provide model reliability

Models mentioned in the paper include:

- EfficientDet variants: EffNetB0 through EffNetB7
- DetRS
- ResNet50
- YOLO

### 3. Coordinator Agent

Expected responsibilities:

- oversee fusion decisions when centralized coordination is used
- resolve conflicts
- finalize merged boxes

### 4. Data Processing Agent

Expected responsibilities:

- preprocess model outputs
- postprocess fused detections
- prepare COCO evaluation files

### 5. Blackboard

The blackboard is the shared communication system.

It should store:

- bounding boxes
- labels
- confidence scores
- model origins
- candidate overlaps
- fusion/competition/negotiation results

---

## Bounding-Box Agent Lifecycle

Each bounding-box agent should follow this lifecycle:

1. **Initializing**
   - created from a detector proposal
   - stores coordinates, confidence, label, source model, and reliability

2. **Exploring**
   - queries the blackboard
   - finds overlapping boxes
   - applies system constraints such as IoU threshold, label consistency, and confidence threshold

3. **Selecting**
   - chooses candidate boxes for interaction
   - checks:
     - spatial overlap
     - confidence
     - category consistency
     - optional model reliability

4. **Coordinating**
   - applies one of:
     - decentralized WBF
     - competition / attack-defense
     - auction-style bid selection
     - multi-round negotiation

5. **Acting**
   - keeps, removes, fuses, or adjusts its box
   - posts final result to the blackboard

---

## Coordinate Format

The code must clearly document and consistently use one bounding-box format.

Acceptable formats:

- `xyxy = [x1, y1, x2, y2]`
- `xywh = [x, y, width, height]`

COCO evaluation normally requires `xywh`.

All IoU, IoB, fusion, and evaluation code must convert formats safely and explicitly.

---

## Required Geometry Metrics

### Intersection over Union

For boxes `A` and `B`:

```text
IoU(A, B) = area(A ∩ B) / area(A ∪ B)


Expected behavior:

identical boxes: IoU = 1
non-overlapping boxes: IoU = 0
partial overlap: value between 0 and 1
Intersection over Box Area

IoB is asymmetric.

IoB(A|B) = area(A ∩ B) / area(A)
IoB(B|A) = area(A ∩ B) / area(B)

This is required for attack-defense competition.

Expected behavior:

if A is fully inside B:
IoB(A|B) = 1
IoB(B|A) < 1
Coordination Mechanism 1: Decentralized WBF / AWBF

The paper says AWBF first reproduces classical WBF behavior in a decentralized multi-agent way.

Given boxes:

B = {b1, b2, ..., bn}
S = {s1, s2, ..., sn}

The fused box is:

bf = Σ wi * bi
wi = si / Σ sj

where si is the confidence score of box i.

Required behavior:

Read boxes, scores, and labels from the blackboard.
Determine overlapping boxes.
Filter candidates using IoU and category consistency.
Apply WBF to selected candidates.
Post fused boxes to the blackboard.

Implementation should preserve classical WBF behavior as much as possible.

Coordination Mechanism 2: Competitive Interaction

Each agent can attack another overlapping box.

For agent Ai attacking box Bj:

S_attack = confidence_Ai * IoB(Bj|Ai)
S_defense = confidence_Bj * IoB(Ai|Bj)
R = S_attack - S_defense

Decision rule:

if R > T:
    Ai wins and Bj is removed
elif R < -T:
    Bj wins and Ai is removed
else:
    Ai and Bj are fused using confidence-weighted average

T is the cooperation threshold.

Interpretation:

T = 1: highly cooperative, close to WBF
T = 0: highly competitive
competitiveness level is approximately 1 - T

Required checks:

competition must only happen between relevant overlapping boxes
category consistency should be enforced
removed boxes must not appear in final output
fused boxes should use confidence-weighted averaging
Coordination Mechanism 3: Auction-Style Winner Determination

The paper also describes competition as auction-style coordination.

Utility:

U(A|B) = α * confidence_A
       + β * IoU(A, B)
       + γ * reliability_A

Bid:

Bid(A|B) = U(A|B) * IoB(B|A)

Winner:

winner = argmax bid

If another box has a close utility score within threshold T, boxes should merge using confidence-weighted averaging.

Required implementation details:

expose alpha, beta, and gamma
define default values
handle missing model reliability
document whether reliability is fixed, learned, or computed from validation metrics
Coordination Mechanism 4: Adaptive Multi-Round Negotiation

Unlike one-shot competition, negotiation runs for multiple rounds.

At each round, agents may:

raise bids
adjust confidence
shift bounding boxes toward stronger competing boxes
fuse if bids are close
stop if no meaningful improvement occurs

Bounding-box update:

B_new = (1 - W) * B_old + W * B_competing

where:

W = (C_prime - C) / (C_prime + C)

C is the current confidence and C_prime is the adjusted or target confidence.

Required parameters:

Rmax: maximum negotiation rounds
W: proposal adjustment weight
T: cooperation/fusion threshold

Interpretation:

Rmax = 1: behaves like one-shot auction
larger Rmax: enables iterative refinement
W = 0: static boxes, auction-like behavior
W → 1: strong movement toward fusion
useful reported range for W: approximately 0.15 to 0.4
beyond about 5 rounds, the paper reports little extra benefit

Termination conditions:

maximum rounds reached
no agent significantly improves its bid
all conflicts resolved
Required Evaluation Dataset

The paper evaluates on COCO-style data.

Required support:

COCO annotations
COCO detection-result JSON
COCO metrics via pycocotools or equivalent

The code must document:

exact COCO split used
whether full COCO or a subset is used
how detector predictions are loaded
how fused predictions are exported
how metrics are computed
Required COCO Metrics

The reproducibility script must report:

Average Precision
AP@[0.50:0.95]
AP@0.50
AP@0.75
AP small
AP medium
AP large
Average Recall
AR@[0.50:0.95]
AR@0.50
AR@0.75
AR small
AR medium
AR large
Reported Paper Results

Codex should try to reproduce these approximately.

Method	AP	AP50	AP75	AP small	AP medium	AP large	AR	AR50	AR75	AR small	AR medium	AR large
WBF	0.673	0.894	0.709	0.605	0.731	0.846	0.471	0.627	0.846	0.800	0.850	0.867
AWBF	0.610	0.660	0.625	0.610	0.766	0.675	0.395	0.676	0.745	0.664	0.706	0.819
AWBF-competition	0.651	0.666	0.590	0.322	0.636	0.632	0.375	0.685	0.764	0.653	0.743	0.840
AWBF-Negotiation	0.626	0.684	0.640	0.622	0.780	0.701	0.413	0.718	0.778	0.684	0.741	0.859
Individual Detector Baselines

The paper reports these detector metrics.

These are useful for checking whether the same prediction files are being used.

Detector	AP	AP50	AP75	AP small	AP medium	AP large	AR	AR50	AR75	AR small	AR medium	AR large
EffNetB0	0.336	0.515	0.354	0.125	0.388	0.528	0.288	0.440	0.467	0.193	0.550	0.688
EffNetB1	0.392	0.581	0.418	0.186	0.447	0.571	0.322	0.501	0.532	0.294	0.599	0.735
EffNetB2	0.425	0.617	0.453	0.238	0.479	0.591	0.340	0.537	0.569	0.347	0.632	0.750
EffNetB3	0.459	0.650	0.491	0.280	0.503	0.616	0.359	0.569	0.604	0.404	0.654	0.770
EffNetB4	0.490	0.685	0.529	0.334	0.538	0.640	0.375	0.598	0.634	0.464	0.682	0.782
EffNetB5	0.505	0.700	0.544	0.343	0.549	0.646	0.383	0.619	0.656	0.500	0.698	0.791
EffNetB6	0.513	0.705	0.555	0.352	0.556	0.652	0.387	0.626	0.664	0.505	0.703	0.795
EffNetB7	0.521	0.710	0.562	0.370	0.562	0.660	0.390	0.633	0.671	0.517	0.711	0.801
DetRS	0.515	0.710	0.654	0.318	0.565	0.676	0.384	0.628	0.671	0.479	0.723	0.828
ResNet50	0.496	0.697	0.538	0.299	0.543	0.656	0.378	0.607	0.640	0.457	0.686	0.800
YOLO	0.500	0.678	0.546	0.336	0.544	0.644	0.381	0.628	0.688	0.533	0.734	0.826
Required Reproducibility Script

Add or fix:

scripts/reproduce_paper_results.py

The script should:

load detector prediction files
run WBF
run AWBF decentralized WBF
run AWBF competition
run AWBF negotiation
export COCO-format prediction JSONs
evaluate with COCO metrics
print a table matching the paper table
save results to outputs/reproduction_report.json

Suggested CLI:

python scripts/reproduce_paper_results.py \
  --annotations data/coco/annotations/instances_val2017.json \
  --predictions-dir data/predictions \
  --output-dir outputs/reproduction \
  --iou-threshold 0.55 \
  --skip-full-coco false

If full data is unavailable, provide a small smoke-test mode:

python scripts/reproduce_paper_results.py --sample
Required Tests

Codex must add or fix tests for the following.

Geometry Tests

File:

tests/test_geometry.py

Required tests:

identical boxes have IoU 1
non-overlapping boxes have IoU 0
partial overlap is correct
IoB is asymmetric
zero-area boxes are handled safely
WBF Tests

File:

tests/test_wbf.py

Required tests:

confidence-weighted average coordinates are correct
boxes with different labels are not fused
boxes below IoU threshold are not fused
fused confidence is deterministic
Competition Tests

File:

tests/test_competition.py

Required tests:

stronger attack removes weaker box
stronger defense keeps attacked box
close scores within threshold cause fusion
threshold T = 1 behaves more cooperatively
threshold T = 0 behaves more competitively
Negotiation Tests

File:

tests/test_negotiation.py

Required tests:

Rmax = 1 behaves like one-shot auction
higher Rmax allows multiple updates
W = 0 keeps coordinates fixed
larger W moves box toward competing box
negotiation terminates when no bid improves
COCO Pipeline Tests

File:

tests/test_coco_eval_pipeline.py

Required tests:

prediction export is valid COCO detection JSON
category IDs are valid
bbox format is xywh
scores are between 0 and 1
small sample evaluation runs without crashing


