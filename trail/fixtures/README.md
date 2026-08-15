# Fixtures: the pinned regression-gate artifacts

These files back `trail reproduce baseline-matrix`. The gate re-scores a fixed set of unlearned
checkpoints and requires every executable row to reproduce its recorded
accuracy panel within ±1 percentage point, so that a library change that
silently alters a metric fails loudly.

## `external_baselines_v1.csv` / `external_baselines_v2.csv`

The pinned ground truth. `v2` is the default used by `reproduce.py`; it is
`v1` plus two external model-card anchor rows (below).

Conventions:

- All accuracy columns are on the **0–100** scale.
- The `ckpt` column holds **file names**, resolved against the directory you
  pass as `--ckpt-root DIR` (CLI) or `ckpt_root=` (API). Rows whose checkpoint
  is not found are reported as `skipped_missing_ckpt`; they do not fail the
  gate. The checkpoints themselves are not redistributed with the library.
- `forget_test_acc` is empty for `random`-mode rows: the forget-class test
  partition is empty by design in that mode.

Each row records what its own checkpoint scores **under this protocol**, which
is what makes the gate a library-fidelity test: it detects drift in TRAIL, not
disagreement with any external publication. Two caveats follow from that and
are worth stating explicitly:

- The `l1sparse_*` rows are self-consistent with their own checkpoints, but the
  sparsity coefficient used to produce them is not the one reported in the
  original L1-Sparse paper. Do not read these rows as a reproduction of
  published L1-Sparse numbers.
- Gold-reference training for `random` mode used `split_seed=0`, which differs
  from the split JSONs below (`split_seed=seed`). Keep this in mind before
  comparing gold-relative numbers against these rows.

### External model-card anchors (in `v2` only)

Two rows come from the public
[jaeunglee/resnet18-cifar10-unlearning](https://huggingface.co/jaeunglee/resnet18-cifar10-unlearning)
checkpoint set (the Unlearning Comparator release, Lee et al., TVCG 2026;
card accessed 2026-07-03). These are the rows a user can actually run without
access to any private checkpoint, since the weights are public:

- `jaeunglee_original` — `full_test_acc=95.4`, the card's CIFAR-10 accuracy.
  Full-test accuracy is class-agnostic, so this row is protocol-exact.
- `jaeunglee_gold_airplane` — `retain_test_acc=95.3`, the card's number for the
  class-excluded retrain. The card does not state the basis of its number;
  measurement disambiguates it as the 9-class accuracy, which is what `ra_test`
  computes under `forget_class=0` (the full-test reading is excluded by ~10 pp).

The set's other per-class golds cannot be scored by this gate, whose protocol
pins `forget_class=0`.

SHA-256:

- `external_baselines_v1.csv` — `48f1fe691b7a96130c92d8c451659dee70312fdbf7834ec5007093e2ae8920f2`
- `external_baselines_v2.csv` — `a0fed49ab89d7ec7deadf22c8b1facef5406409dea80fdddaf44d607be9b181f`

## `splits/forget_{single_class,random}_seed{2,3,42}.json`

The exact forget-set indices used by the fixture rows, so that a re-run
partitions the data identically. Protocol: a 45k/5k per-class train/validation
carve-out drawn from `numpy.random.RandomState(carveout_seed)`,
augmentation-stripped evaluation, batch size 256.

## `reference_logits.npz` / `reference_checksums.json`

A triage oracle for environment drift: the logits of the first eight canonical
test images under one fixture checkpoint, together with the summary statistics
of the transformed input. If these do not reproduce, the problem is in the
preprocessing or model-construction stack rather than in a metric, and should
be debugged there before interpreting a gate failure.
