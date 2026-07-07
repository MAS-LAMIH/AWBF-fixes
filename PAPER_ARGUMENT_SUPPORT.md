# Paper Argument Support

This file classifies the paper claims from the prompt against the current code.
Because the full benchmark run could not complete in this environment, verdicts
that depend on reproduced metrics are conservative.

| # | Paper claim | Paper metric evidence | Current-code evidence | Verdict | Can revised paper keep it? | Suggested wording |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | WBF is strongest overall baseline. | WBF has highest AP/AP50/AP75 and strong AR in the paper table. | WBF is implemented as confidence-weighted cluster fusion; full metrics not computed here. | CANNOT DETERMINE FROM CURRENT RUN | Yes, if reproduced metrics confirm. | "On the reported benchmark, WBF remains the strongest centralized baseline." |
| 2 | AWBF is comparable but not better than WBF and mainly decentralizes WBF. | Paper AWBF has lower AP than WBF but some useful metric preservation. | Current AWBF uses the same fusion path as WBF, so equivalence is expected. | PARTIALLY SUPPORTED | Yes, but emphasize formulation not improvement. | "Current implementation treats AWBF as an agent-style execution of WBF; it should match WBF unless additional agent behavior is introduced." |
| 3 | AWBF improves/preserves AP small, AP medium, AR50 vs WBF. | Paper table gives AP small/medium and AR50 improvements. | Current code likely makes AWBF identical to WBF, so it cannot explain improvements over WBF. | NOT SUPPORTED BY CURRENT CODE | Only if metrics from a different AWBF implementation support it. | "The reported AWBF gains are empirical for the paper setup; this code's AWBF is equivalent to WBF." |
| 4 | Competition can improve precision as competitiveness increases. | Paper interpretation links suppression to precision. | Competition code removes a loser when attack-defense margin exceeds threshold. | PARTIALLY SUPPORTED | Yes as a mechanistic hypothesis, not guaranteed metric result. | "Competition is designed to be precision-oriented by suppressing weaker overlaps; the AP effect is parameter/data dependent." |
| 5 | Competition can harm small-object performance. | Paper AP small drops strongly for competition. | Suppression of weak overlaps can remove valid small objects; small IoU is sensitive to coordinate noise. | PARTIALLY SUPPORTED | Yes with caveat. | "Competition may harm small objects because suppression errors are costlier at small scale." |
| 6 | Negotiation restores adaptability vs competition. | Paper negotiation improves AP small/medium/large and AR metrics vs competition. | Negotiation moves/fuses instead of immediately deleting every weak box. | PARTIALLY SUPPORTED | Yes if reproduced metrics confirm. | "Negotiation is intended to reduce premature suppression; measured gains should be reported from validation runs." |
| 7 | Negotiation avoids premature suppression. | Paper interpretation. | Code performs iterative proposal adjustment/fusion before final output. | SUPPORTED MECHANISTICALLY | Yes. | "Negotiation delays hard suppression by allowing multi-round adjustment and fusion." |
| 8 | Negotiation benefits small/medium precision, AR50, large AR. | Paper table. | Mechanism is plausible; current full metrics unavailable. | CANNOT DETERMINE FROM CURRENT RUN | Only with computed metrics. | "In the reported table, negotiation improves these metrics over competition; reproduce with validation before generalizing." |
| 9 | Parameter trends for T, W, Rmax. | Paper summary describes empirical ranges. | Parameter sweep script is provided to test T/W/Rmax; results not computed here. | CANNOT DETERMINE FROM CURRENT RUN | Keep as empirical if sweep supports. | "Parameter effects are empirical and should be shown with sweep plots/tables." |

## Algorithmic explanations for reported trends

### A. Why WBF can have highest overall AP

WBF aggregates multiple detector boxes into confidence-weighted representatives,
which can reduce duplicate false positives and improve localization when detectors
agree. This can raise AP and high-IoU AP because the fused box is often closer to
consensus than a single detector output.

### B. Why AWBF can be comparable but lower than WBF

A decentralized/agentified implementation can reproduce WBF if agents select the
same clusters and weights. If agent ordering, candidate visibility, or thresholds
differ, the result can become comparable but lower. The current code does not add
such behavior, so it should be equivalent to WBF.

### C. Why AWBF may improve AP small/medium or AR50 while lowering AP

This is plausible only if AWBF preserves some detections or clusters differently
from centralized WBF. Preserving extra small/medium detections can improve some
scale-specific or low-IoU recall metrics while adding false positives or worse
localization that lowers overall AP. The current code's AWBF path does not create
this distinction from WBF.

### D. Why competition can increase precision but harm recall/small objects

Competition removes a box when attack-defense margin exceeds threshold. Removing
weak overlapping boxes can reduce false positives and improve precision/AP, but
it can also remove true positives. Small objects are especially vulnerable because
minor coordinate changes cause large IoU changes.

### E. Why negotiation can restore metrics compared with competition

Negotiation updates boxes toward competing proposals and can fuse when utilities
are close. This avoids immediate deletion and can refine positions, which may
recover small/medium/large AP and AR compared with pure suppression.

### F. Why Rmax > 5 can show diminishing returns

If boxes converge within a few proposal updates, extra rounds repeatedly compare
similar boxes and add little new information. Additional rounds can increase cost
without changing cluster membership or coordinates much.

### G. Why moderate W may be better than W=0 or W close to 1

`W=0` means boxes do not move, making negotiation close to a static auction.
Moderate `W` allows gradual correction. `W` close to 1 can over-move boxes toward
a competitor, potentially hurting localization or causing unstable convergence.

### H. Whether W≈0.7 metric convergence is explainable

It is plausible as an empirical balance point where boxes move strongly enough to
reach consensus but not enough to destroy localization. However, this is not a
mechanistic guarantee. It must be validated with parameter sweep results.

## Current overall conclusion

The code can justify some **mechanistic explanations** in the paper, especially
for competition and negotiation. It cannot currently justify paper numeric trends
without a successful validated benchmark run. Claims that AWBF differs
behaviorally from WBF should be revised unless additional non-equivalent AWBF
agent behavior is implemented and validated.
