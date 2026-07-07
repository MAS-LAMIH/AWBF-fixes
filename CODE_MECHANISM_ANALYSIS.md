# Code Mechanism Analysis

This is an **analysis-only** audit. It uses the source code and the paper-result
summary supplied in the prompt. It does **not** claim that the current repository
has reproduced the paper numbers, because no full benchmark/COCO evaluation was
run for this audit.

## Evidence inspected

- `wbf_agents/awbf.py`: WBF/AWBF primitives, competition, negotiation, bbox
  conversion, and COCO evaluation.
- `scripts/reproduce_paper_results.py`: method selection and per-method output
  generation.
- Existing generated report files, where present, are treated as repository
  artifacts rather than new full-dataset evidence.

## Metric terminology check

The code no longer treats standard COCOeval `stats[7]` and `stats[8]` as AR50 and
AR75. Standard COCOeval metrics are reported as `AR_maxDets1`, `AR_maxDets10`,
and `AR`/`AR_maxDets100`. The code separately computes custom `AR50` and `AR75`
from the COCOeval recall tensor. Therefore, comparisons to paper AR50/AR75 are
valid only when those custom fields are populated by an actual evaluation run.

## Method-by-method mechanism analysis

### WBF

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | The implementation collects detections by model, filters by score, sorts by confidence, assigns same-label boxes to IoU-overlapping clusters, and emits confidence-weighted representative boxes. |
| Expected effect on false positives | Reduces duplicate overlapping predictions from multiple detectors, which can lower duplicate false positives. |
| Expected effect on false negatives | Usually preserves consensus detections, but weak alternatives can be absorbed into a cluster rather than emitted separately. |
| Expected effect on AP | Plausibly strong, because duplicate suppression plus confidence-weighted localization can improve precision and high-IoU localization. |
| Expected effect on AR | Can be strong when detector consensus is good, but not necessarily maximal if extra low-confidence alternatives are merged or filtered. |
| Small/medium/large effects | Stable for medium/large objects where detector agreement is easier. Small objects remain sensitive to small coordinate shifts. |
| Does it plausibly explain the paper result? | Yes, mechanistically. WBF having the strongest overall AP is consistent with confidence-weighted consensus fusion, but the numeric claim still requires a dataset run. |

### AWBF

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | In the current reproduction script, `AWBF` uses the same fusion path as `WBF`: with pre-clustering it calls the same cluster-fusion helper; without pre-clustering it calls `decentralized_wbf()` just like WBF. |
| Expected effect on false positives | Same as WBF in this implementation. |
| Expected effect on false negatives | Same as WBF in this implementation. |
| Expected effect on AP | Should be identical or nearly identical to WBF unless future code adds non-equivalent blackboard/agent ordering behavior. |
| Expected effect on AR | Should be identical or nearly identical to WBF. |
| Small/medium/large effects | Should match WBF. |
| Does it plausibly explain the paper result? | It explains AWBF as a decentralized/execution-style reformulation of WBF. It does **not** explain paper-table AWBF metrics that differ from WBF, such as higher AP small/AP medium/AR50 but lower overall AP. Those differences require either a different AWBF implementation or empirical effects not present in this code path. |

### Incremental_AWBF

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | Uses WBF-style cluster membership but updates cluster representatives incrementally as detections arrive. It is an agent-style execution variant, not a separate paper method unless explicitly reported as an ablation. |
| Expected effect on false positives | If cluster membership is fixed, same as WBF/AWBF. |
| Expected effect on false negatives | If cluster membership is fixed, same as WBF/AWBF. |
| Expected effect on AP | Should match AWBF/WBF except for order effects or floating-point differences. |
| Expected effect on AR | Should match AWBF/WBF except for order effects or floating-point differences. |
| Small/medium/large effects | No distinct scale-specific effect is implied by the mechanism alone. |
| Does it plausibly explain the paper result? | No as an independent paper result. It is useful for testing decentralized execution equivalence and order sensitivity. |

### AWBF-competition

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | Same-label overlapping pairs are compared with attack and defense scores based on confidence and IoB. If the score margin exceeds the threshold, one box survives; otherwise the pair is confidence-weighted fused. |
| Expected effect on false positives | Can reduce false positives by removing weaker overlapping boxes, especially under more competitive/lower-threshold settings. |
| Expected effect on false negatives | Can increase false negatives when a suppressed lower-confidence box was actually a valid true positive, particularly in crowded scenes or for small objects. |
| Expected effect on AP | Can improve precision/AP if the removed boxes are mostly duplicate or false-positive boxes; can hurt AP if true positives are removed. |
| Expected effect on AR | May reduce AR because hard suppression removes candidate detections. The paper claim that recall remains relatively stable is plausible but data-dependent. |
| Small/medium/large effects | Small objects are especially vulnerable: small coordinate errors have larger IoU consequences, and weaker small-object detections may be removed. Medium/large objects may tolerate suppression better if detector consensus is stronger. |
| Does it plausibly explain the paper result? | Yes for the qualitative trend: competition can improve precision while harming small-object performance. The exact AP/AR balance must be empirically verified. |

### AWBF-Negotiation

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | Same-label overlapping boxes enter a multi-round bid/proposal process. If utilities are close, boxes are fused; otherwise one proposal can move toward the competing box with `B_new = (1-W) * B_old + W * B_competing`, and score is adjusted. |
| Expected effect on false positives | Can reduce duplicate false positives through fusion while being less abrupt than hard competition. |
| Expected effect on false negatives | Can reduce false negatives relative to pure competition by avoiding immediate removal and allowing iterative refinement/fusion. |
| Expected effect on AP | Can recover AP relative to pure competition when refinement improves localization or avoids deleting true positives. Excessive movement can hurt localization. |
| Expected effect on AR | Can recover AR relative to competition because more detections can survive as refined/fused proposals rather than being prematurely suppressed. |
| Small/medium/large effects | Small/medium/large performance can improve relative to competition when negotiation preserves boxes that competition would delete. Small objects can still be harmed if movement creates localization error. |
| Does it plausibly explain the paper result? | Yes, mechanistically, for negotiation outperforming competition on small/medium/large metrics and AR. Numeric support still requires evaluation. |

### AWBF-competition-IncrementalState and AWBF-Negotiation-IncrementalState

| Aspect | Analysis |
| --- | --- |
| Implemented behavior | Experimental variants that mutate or collapse cluster state incrementally after competition/negotiation behavior. They are emitted as separate outputs and should not overwrite paper-style competition/negotiation outputs. |
| Expected effect on false positives | Can remove more boxes than paper-style variants if state collapse is aggressive. |
| Expected effect on false negatives | Can increase false negatives by reducing candidate diversity. |
| Expected effect on AP | Data-dependent; can be much lower if over-collapsing removes useful detections. |
| Expected effect on AR | Likely lower than non-incremental-state variants when many detections are collapsed/removed. |
| Small/medium/large effects | Small objects are likely most sensitive to aggressive state collapse. |
| Does it plausibly explain the paper result? | No. These are ablation/experimental variants, not paper-style methods. |

## Required algorithmic explanations

### A. Why WBF has strongest overall AP

WBF is a strong AP baseline because it suppresses duplicate overlapping outputs
while using confidence-weighted averaging to improve localization. If the 21
input detector files contain many correlated detections around the same true
objects, consensus fusion can reduce false positives and improve high-IoU match
quality at the same time. This mechanism is consistent with WBF leading AP,
AP50, and AP75 in the supplied paper table.

### B. Why AWBF can be comparable but lower than WBF

A decentralized AWBF formulation can be comparable to WBF when agents see the
same candidates and apply the same confidence-weighted fusion rule. It can become
lower than centralized WBF if agent ordering, blackboard visibility, local
candidate selection, or thresholds differ from the centralized implementation.
However, the current code does not introduce such a difference for the `AWBF`
method; it routes AWBF through the same fusion implementation as WBF. Therefore,
this repository currently supports AWBF-as-equivalent-execution-style, not a
behaviorally lower AWBF variant.

### C. Why AWBF can improve AP small, AP medium, or AR50 while lowering overall AP

This trend is plausible only for an AWBF implementation that differs from WBF.
For example, if decentralized agents preserve additional small/medium proposals
or use local cluster decisions that retain more low-IoU true positives, AP small,
AP medium, or AR50 can improve while overall AP falls because extra detections
also increase false positives or reduce high-IoU localization quality. The
current code path for AWBF does not create that distinction, so this explanation
is theoretical relative to the supplied paper summary rather than demonstrated by
this implementation.

### D. Why competition can improve AP/precision but harm small-object performance

Competition makes pairwise hard decisions: when attack sufficiently exceeds
defense, one overlapping box is removed. If removed boxes are mostly duplicates
or false positives, precision and AP can improve. If removed boxes are true
positives, recall drops. Small objects are most fragile because a few pixels of
coordinate error can change IoU substantially, and weak small-object detections
may be suppressed even when valid. This directly matches the paper interpretation
of improved precision with possible small-object degradation.

### E. Why negotiation can recover small/medium/large performance compared with competition

Negotiation avoids immediate hard deletion by allowing boxes to move toward a
competing proposal or fuse when bids are close. That can preserve useful evidence
that pure competition would discard and can refine localization before final
output. This mechanism plausibly explains recovery in AP small/medium/large and
AR relative to competition in the supplied paper table.

### F. Why Rmax beyond about 5 may give diminishing returns

The negotiation update is iterative but local to overlapping box pairs. Once a
pair has fused, converged, or stopped improving its bid, additional rounds add
repeated comparisons with little new information. Therefore, a small number of
rounds can capture most of the adjustment, while larger `Rmax` increases runtime
more than it changes outputs. This is a mechanism-based explanation; the exact
"5 rounds" point is empirical.

### G. Why moderate W can outperform W = 0 or W close to 1

`W = 0` prevents coordinate movement, reducing negotiation to a static auction or
fusion decision. Moderate `W` allows gradual correction toward a competing
proposal while retaining the original detection geometry. `W` close to 1 moves
boxes almost entirely to the competing proposal, which can over-correct,
collapse diversity, or hurt localization. Thus the useful reported range
`W = 0.15–0.4` is plausible from the update rule, but the optimum must be
validated empirically.

### H. Whether W around 0.7 metric convergence is explainable from code

The code can suggest a qualitative reason: high `W` strongly pulls proposals
toward each other, so different metrics may converge if outputs collapse toward a
common consensus. But the exact claim that `W ≈ 0.7` causes metric convergence is
not guaranteed by the implementation. It should be treated as an empirical
observation that requires parameter-sweep evidence.

## Current overall conclusion

- The code can explain the **mechanisms** behind WBF consensus strength,
  competition's precision/suppression tradeoff, and negotiation's potential to
  recover detections relative to pure competition.
- The current `AWBF` implementation is WBF-equivalent in behavior, so it cannot
  by itself justify paper-table AWBF differences from WBF.
- The code includes custom AR50/AR75 computation, but AR50/AR75 claims require a
  successful evaluation run that actually populates those fields.
- Paper numeric claims should be presented as reported empirical results, not as
  reproduced by this repository, until a full validated benchmark run is
  available.
