# Code Mechanism Analysis

This analysis uses the paper summary supplied in the prompt as the paper target.
It separates (1) what the current code mechanically implements from (2) what can
be concluded from computed metrics. In this environment the full benchmark/COCO
run could not complete because the expected benchmark CSVs and COCO annotations
are absent, so metric-based claims remain conditional on a successful local run.

## WBF

- **Implemented mechanism:** `decentralized_wbf()` performs score-sorted WBF by
  grouping same-label boxes whose IoU with a confidence-weighted cluster
  representative exceeds the threshold, then emits confidence-weighted fused
  boxes.
- **Expected AP effect:** high precision/localization when detector boxes agree;
  duplicate detections are reduced.
- **Expected AR effect:** can reduce duplicate/noisy boxes, but may also lower
  recall if valid low-confidence alternatives are absorbed or filtered.
- **Small/medium/large expectation:** small objects are sensitive to coordinate
  shifts, but WBF is usually stable when multiple detectors agree.
- **Matches paper interpretation:** partially from mechanism. The paper's claim
  that WBF is the strongest baseline requires computed metrics.
- **Current-code evidence:** implemented as the baseline fusion path for `WBF`.

## AWBF

- **Implemented mechanism:** in the current reproduction script, `AWBF` uses the
  same WBF cluster/reduction path as `WBF`; it is an execution-style label rather
  than a behaviorally different algorithm.
- **Expected AP effect:** identical or nearly identical to WBF in current code.
- **Expected AR effect:** identical or nearly identical to WBF in current code.
- **Small/medium/large expectation:** identical or nearly identical to WBF in
  current code.
- **Matches paper interpretation:** only the decentralized-equivalence argument
  is supported. Any paper-table difference between WBF and AWBF is not explained
  by this current implementation unless additional agent/blackboard behavior is
  added.
- **Current-code evidence:** method-equivalence reporting should classify WBF vs
  AWBF as identical/nearly identical when both are selected.

## Incremental_AWBF

- **Implemented mechanism:** uses the same WBF assignment logic as AWBF/WBF but
  maintains each cluster representative as incremental running state.
- **Expected AP/AR effect:** identical/nearly identical to AWBF for fixed WBF
  membership, except for floating-point noise.
- **Small/medium/large expectation:** identical/nearly identical to AWBF.
- **Matches paper interpretation:** this is an implementation/execution variant,
  not a separate paper metric unless explicitly reported as such.

## AWBF-competition

- **Implemented mechanism:** overlapping same-label detections are compared with
  attack/defense scores based on confidence and IoB. If the margin exceeds the
  threshold, one box survives; otherwise the pair is fused.
- **Expected AP effect:** may improve precision by removing weaker or duplicate
  boxes, especially at lower cooperation threshold `T`.
- **Expected AR effect:** may reduce recall if valid detections are suppressed,
  especially small or weak objects.
- **Small/medium/large expectation:** small-object AP/AR can degrade because small
  coordinate errors have large IoU impact and weak true positives can be removed.
- **Matches paper interpretation:** mechanism supports the qualitative explanation
  that competition can be more suppressive. Whether AP increases or AR remains
  stable must be verified with computed metrics.

## AWBF-Negotiation

- **Implemented mechanism:** overlapping same-label boxes run a multi-round
  negotiation. Each round compares auction-style utilities and can move a box
  toward the competing proposal using `B_new=(1-W)B_old+W B_competing`, with score
  adjustment.
- **Expected AP effect:** can improve localization/precision relative to pure
  competition when boxes converge instead of being immediately removed.
- **Expected AR effect:** can preserve more detections than competition when it
  fuses/refines rather than suppresses.
- **Small/medium/large expectation:** can restore small/medium/large metrics
  relative to competition if it avoids premature removal; excessive movement can
  hurt localization.
- **Matches paper interpretation:** mechanism supports the negotiation story, but
  support for the paper's numeric trend requires computed metrics.

## Incremental-state competition/negotiation variants

- **Implemented mechanism:** these are experimental variants that collapse
  post-competition or post-negotiation cluster state incrementally.
- **Expected AP/AR effect:** may differ substantially from paper-style outputs;
  they should not be used as paper metrics unless explicitly reported.
- **Matches paper interpretation:** not paper-style methods; useful only as
  ablations.

## Bbox export/evaluation

- **Implemented mechanism:** exports COCO detection JSON, converts normalized
  boxes to pixel-space `xywh` when image sizes are available, validates image IDs,
  category IDs, score ranges, and bbox validity, and removes invalid rows before
  evaluation.
- **Metric naming:** COCO standard `stats[7]`/`stats[8]` are now reported as
  `AR_maxDets10` and `AR`/`AR_maxDets100`; custom `AR50`/`AR75` are computed from
  the COCOeval recall tensor and should not be confused with standard stats.

## Overall verdict

- The code supports explaining **why** WBF, competition, and negotiation might
  have the qualitative effects described by the paper.
- The code does **not** by itself prove the paper's numeric trends until the full
  benchmark/COCO run succeeds and the generated reports show nonzero, validated
  metrics.
- In the current implementation, WBF/AWBF/Incremental_AWBF should be described as
  equivalent execution styles, not as independent behaviorally different methods.
