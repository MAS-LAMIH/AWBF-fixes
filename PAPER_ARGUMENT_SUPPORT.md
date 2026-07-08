# Paper Argument Support

This document classifies the paper claims from the prompt against the current
source code. It is intentionally conservative: this audit did **not** run the
full benchmark, COCO evaluation, or parameter sweep.

## Verdict labels

- **SUPPORTED BY CODE MECHANISM**: the implemented algorithm directly contains a
  mechanism that supports the claim qualitatively.
- **PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION**: the mechanism could produce the
  claim, but the claim depends on dataset metrics or parameter settings.
- **NOT SUPPORTED BY CURRENT CODE**: the current implementation does not contain
  the mechanism needed for the claim, or implements behavior that contradicts it.
- **CANNOT DETERMINE WITHOUT DATASET RUN**: the claim is primarily numeric and
  requires full benchmark/COCO evaluation or a parameter sweep.

## Claim-by-claim support table

| # | Paper claim | Paper metric evidence from prompt | Current-code evidence | Verdict | Can the revised paper keep it? | Suggested revised wording |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | WBF is the strongest overall baseline. | WBF has the highest reported AP, AP50, AP75, and strong scale AR metrics. | WBF uses confidence-weighted consensus fusion, which can reduce duplicates and improve localization. | PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION | Yes, as a reported empirical result; not as a claim reproduced here unless a validated run confirms it. | "On the reported COCO subset, WBF remains the strongest centralized baseline; this is consistent with confidence-weighted consensus fusion." |
| 2 | AWBF is comparable to WBF and mainly decentralizes WBF. | Paper AWBF is lower overall than WBF but remains competitive. | Current AWBF uses the same fusion path as WBF, so equivalence/comparability is expected. | SUPPORTED BY CODE MECHANISM | Yes, if framed as an execution/formulation claim. | "AWBF can be interpreted as an agent-style execution of WBF; when agents select the same clusters and weights, it should reproduce WBF behavior." |
| 3 | AWBF improves AP small, AP medium, or AR50 over WBF while lowering overall AP. | Prompt reports AWBF APs/APm/AR50 higher than WBF while overall AP is lower. | Current AWBF is behaviorally identical to WBF, so it does not explain improvements over WBF. | NOT SUPPORTED BY CURRENT CODE | Only if supported by a different/non-equivalent AWBF implementation or full empirical results. | "The reported AWBF scale-specific gains should be treated as empirical for the paper setup; the current code's AWBF path is WBF-equivalent." |
| 4 | Competition improves precision as competitiveness increases. | Paper interpretation says suppression removes weaker boxes and can increase AP. | Competition computes attack/defense margins and removes a loser when the threshold condition is met. | PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION | Yes as a mechanism-backed hypothesis; AP improvement must be measured. | "Competition is designed to be precision-oriented by suppressing weaker overlaps; the measured AP effect is parameter- and data-dependent." |
| 5 | Competition can harm small-object performance. | Prompt reports a large AP-small drop for competition. | Hard suppression can remove weak true positives; small-object IoU is highly sensitive to small coordinate errors. | SUPPORTED BY CODE MECHANISM | Yes, with empirical results cited separately. | "The competition rule can disproportionately harm small objects because erroneous suppression and small coordinate shifts have larger IoU consequences." |
| 6 | Competition preserves recall relatively well because enough accurate boxes survive. | Prompt reports competition AR lower than WBF/AWBF-Negotiation but AR50/AR75 remain nonzero/competitive. | Code can preserve winners and fuse close-margin pairs, but can also delete boxes. | PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION | Only with measured AR trends. | "Recall may remain stable when surviving boxes are accurate, but this must be verified for each threshold and dataset." |
| 7 | Negotiation restores adaptability compared with pure competition. | Prompt reports negotiation improves APs/APm/APl and AR metrics over competition. | Negotiation adjusts or fuses boxes instead of immediately applying hard suppression. | SUPPORTED BY CODE MECHANISM | Yes as a mechanism claim; metric magnitude requires validation. | "Negotiation reduces premature suppression by allowing iterative adjustment and fusion before final output." |
| 8 | Negotiation is especially beneficial for small/medium precision, AR50, and large-object AR. | Prompt reports improvements in those metrics over competition. | The mechanism can preserve/refine boxes, but exact scale-specific gains depend on data and parameters. | PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION | Yes if backed by actual evaluation tables. | "The observed metrics are consistent with negotiation preserving useful proposals; the scale-specific trend should be validated empirically." |
| 9 | Useful W range is around 0.15-0.4. | Paper summary reports this range. | The update rule makes moderate movement plausible, but no source-only proof identifies this optimum. | CANNOT DETERMINE WITHOUT DATASET RUN | Keep only as empirical. | "In the reported sweep, moderate W values performed best; this trend requires empirical validation for new detector sets." |
| 10 | Rmax = 5 gives useful adaptation and Rmax = 10 adds limited gain. | Paper summary reports diminishing returns. | The iterative pair update can converge or stop improving after few rounds, but the exact round count is empirical. | PLAUSIBLE BUT NEEDS EMPIRICAL CONFIRMATION | Yes with sweep evidence. | "Additional rounds may have diminishing returns once pair proposals converge; the practical cutoff should be shown by a sweep." |
| 11 | W around 0.7 may lead to metric convergence. | Paper summary treats this as an empirical observation. | High W can pull boxes strongly toward consensus, but no deterministic guarantee exists. | CANNOT DETERMINE WITHOUT DATASET RUN | Only as cautious empirical observation. | "Metric convergence near W≈0.7, if observed, should be presented as empirical rather than guaranteed by the algorithm." |
| 12 | Current code reproduces the paper metrics. | Paper table provides targets. | This audit did not run full evaluation; known repository reports are conservative about missing data/annotations. | CANNOT DETERMINE WITHOUT DATASET RUN | No, not from this audit. | "The implementation provides the machinery for reproduction; exact reproduction requires the benchmark CSVs, COCO annotations, and validated evaluation outputs." |

## Interpretation of the revised paper argument

The revised paper can safely keep mechanism-based claims that are directly
implemented: WBF performs confidence-weighted consensus fusion; competition uses
confidence/IoB attack-defense suppression; negotiation uses multi-round proposal
adjustment and fusion. These mechanisms justify cautious language such as
"the mechanism suggests" or "the observed metrics are consistent with."

The revised paper should be more careful with claims that depend on numeric
trends. In particular, AWBF-vs-WBF differences are **not supported by the current
code path**, because the current AWBF method is routed through the same fusion
implementation as WBF. If the paper wants AWBF to have distinct AP/AR behavior,
it must document the non-equivalent agent behavior that produces different
cluster membership, ordering, filtering, or score updates.

## Recommended cautious wording

- "The mechanism suggests that competition can improve precision by suppressing
  weaker overlapping detections, but the AP/AR tradeoff is threshold-dependent."
- "The observed negotiation metrics are consistent with avoiding premature
  suppression through iterative proposal adjustment."
- "AWBF is best described as a decentralized execution of WBF when it uses the
  same candidate selection and confidence-weighted fusion rule."
- "Scale-specific gains such as AP small, AP medium, and AR50 require empirical
  validation and should not be inferred from the current WBF-equivalent AWBF code
  path alone."
- "The W and Rmax trends should be reported as parameter-sweep findings rather
  than deterministic properties of the algorithm."

## Metric terminology note

The implementation distinguishes standard COCO recall statistics from custom
AR-at-IoU metrics. Standard `stats[7]` and `stats[8]` are not AR50/AR75; the code
reports them as `AR_maxDets10` and `AR`/`AR_maxDets100` and computes custom
`AR50`/`AR75` from the recall tensor. Therefore, any paper comparison involving
AR50 or AR75 must use those custom fields from a validated COCO evaluation run.
