# Benchmark reproduction performance report

## Investigation summary

The observed `fusion progress: 1/5000 images` behavior is **not a deadlock** in
the current code path. It is consistent with extremely slow Python execution in
nested pairwise box-comparison loops.

The traceback location reported by the user matches the hot path:

```text
awbf_negotiation(...)
  -> nested pair scan over remaining boxes
  -> iou(...)
  -> intersection_area(...)
```

The reproduction script now includes `--profile`, which prints per-image box
counts, estimated pair comparisons, per-negotiation-pass comparison counts,
elapsed time, ETA, and cumulative per-strategy timings.

## Profile command

```bash
python scripts/reproduce_paper_results.py \
  --benchmark-dir data/benchmark \
  --output-dir outputs/reproduction_benchmark \
  --profile \
  --progress-interval 1
```

## Why image 1 can appear stalled

The reported first image has about 742 boxes. A single full pairwise scan costs:

```text
742 * 741 / 2 = 274,911 pair comparisons
```

Each comparison may call `iou(...)`, which calls `intersection_area(...)` and box
area calculations. AWBF competition and negotiation repeat pair scans after each
merge/removal. If many boxes overlap and each pass changes only one pair, the
number of scans can grow with the number of boxes.

## Asymptotic complexity

Let:

- `I` = number of images
- `M` = number of detector/model CSV files
- `n_i` = boxes for image `i` across all models
- `k_i` = clusters/fused boxes retained during WBF-style matching
- `Rmax` = max rounds inside `negotiate_pair` (default 5)

### WBF / AWBF decentralized WBF

The current pure-Python WBF-style implementation scans clusters for each box:

```text
per image: O(n_i * k_i), worst case O(n_i^2)
total benchmark: O(sum_i n_i * k_i), worst case O(sum_i n_i^2)
```

### AWBF-Competition

The competition loop scans box pairs, removes/fuses at most one pair, then
restarts scanning. Worst-case behavior can be cubic when many passes are needed:

```text
single pass: O(n_i^2)
worst case passes: O(n_i)
per image worst case: O(n_i^3)
total benchmark worst case: O(sum_i n_i^3)
```

### AWBF-Negotiation

`awbf_negotiation` has the same outer restart-on-change pair scan as competition.
When an overlapping pair is found, it calls `negotiate_pair`, which performs up
to `Rmax` bid/update rounds. Each bid uses IoU/IoB calculations.

```text
single outer pair scan: O(n_i^2)
per pair negotiation: O(Rmax)
worst case passes: O(n_i)
per image worst case: O(n_i^3 + merges * Rmax)
approx worst case: O(n_i^3 + n_i * Rmax)
total benchmark worst case: O(sum_i n_i^3 + sum_i n_i * Rmax)
```

Because `Rmax` defaults to 5, the repeated pair scans dominate. In practice,
this makes the algorithm genuinely running but potentially extremely slow.

## Benchmark-scale estimate

The user's observed benchmark load is:

```text
21 detector CSV files
978,214 detections
5,000 images
average boxes/image ~= 195.6
```

A single full pairwise scan at the average image size costs approximately:

```text
195.6 * 194.6 / 2 ~= 19,030 comparisons per image
19,030 * 5,000 ~= 95 million comparisons per one full scan across the benchmark
```

This is only the cost of one pairwise scan. Competition and negotiation can
restart scans many times per image, so total comparisons may be much larger. For
an image with 742 boxes, a single full scan is 274,911 comparisons; repeated
passes can quickly reach tens or hundreds of millions of Python-level operations
for that one image.

## Slowest functions / hot spots

Expected hot spots from code inspection and the user's Ctrl+C traceback:

1. `awbf_negotiation(...)` outer nested pair scan.
2. `awbf_competition(...)` outer nested pair scan.
3. `iou(...)` and `intersection_area(...)`, called for most candidate pairs.
4. `auction_bid(...)` / `negotiate_pair(...)`, called after a candidate pair is
   found and adding up to `Rmax` rounds of IoU/IoB work.
5. `decentralized_wbf(...)` cluster matching, worst-case quadratic per image.

## Timing instrumentation added

`--profile` now reports:

- image index and total image count
- image id
- total boxes in the image
- pair comparisons per full scan
- negotiation pass number
- cumulative negotiation comparisons
- elapsed time
- ETA
- cumulative timing for:
  - WBF
  - AWBF
  - AWBF-Competition
  - AWBF-Negotiation

The JSON report now includes `timings_seconds` for those four fusion strategies.

## Recommendations (not implemented yet)

Because competition and negotiation are O(n²) per scan and can become O(n³) per
image with repeated restarts, optimize before attempting full benchmark runs:

1. **Group by label before pairwise matching** so different categories are never
   compared.
2. **Spatial indexing / grid bucketing / R-tree** to avoid IoU calls for boxes
   that cannot overlap.
3. **Vectorize IoU calculations** with NumPy for per-label box arrays.
4. **Limit candidates per box** by score threshold, top-k per label/image, or
   detector-specific prefiltering.
5. **Avoid restart-from-zero after each merge**; maintain an adjacency graph or
   union-find clusters of overlapping boxes.
6. **Add a max comparisons/images guard** for interactive runs.
7. **Run one strategy at a time** via a future `--methods` flag to avoid running
   all four fusion methods when debugging performance.
8. **Parallelize per-image fusion**, since images are independent.

## Conclusion

The benchmark reproduction is almost certainly **extremely slow rather than
stuck**. The first image can be expensive enough to appear frozen because
AWBF-Negotiation performs repeated pure-Python pairwise IoU scans. Use
`--profile` to confirm time and comparison counts before implementing the
optimizations above.
