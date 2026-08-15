# TRAIL

TRAIL is a unified, reproducible evaluation library for machine unlearning. It scores *unlearned checkpoints* — it implements no unlearning methods — under one frozen, modality-agnostic protocol: a single `evaluate()` contract, per-example bootstrap uncertainty on every number, machine-readable skip reasons for everything it cannot compute, full provenance (checkpoint hashes, dataset fingerprints, hidden-preprocessing manifest) on everything it can, and a shipped regression gate that ties each release back to published baseline numbers. Its attack axes are standardized membership-inference and relearning (fine-tuning recovery) evaluations.

## Install

```bash
pip install -e .            # from the repository root
pip install -e .[tracking]  # optional: Weights & Biases logging
```

## Quickstart

```python
from trail import evaluate
from trail.data.specs import DatasetSpec

report = evaluate(
    data=DatasetSpec(mode="single_class", forget_class=0,
                     carveout_seed=2, data_dir="./data"),
    seed=2,
    checkpoints={"unlearned": "ckpts/unlearned_seed2.pth"},  # + original=, gold=
    metrics="default",
)
print(report.to_markdown())
report.to_json("runs/method_seed2.json")
```

Multi-seed comparison is an aggregator concern (project convention: seeds {2, 3, 42}); the per-call API takes exactly one seed.

## CLI

| Command | Purpose |
|---|---|
| `trail eval --config cfg.yaml [-o key=val] [--out report.json] [--markdown]` | run one evaluation request from YAML + dotted overrides |
| `trail aggregate runs/*.json -o results.csv [--labels labels.yaml]` | merge reports into the multi-seed CSV (the only CSV writer) |
| `trail compare runs/*.json --baseline NAME --metric NAME` | paired significance test across the seeds two methods share |
| `trail reproduce baseline-matrix [--ckpt-root DIR] [--tolerance 1.0]` | the G10 regression gate against the pinned fixture |

`python -m trail` works as an alias for the `trail` script. A starting config
is in [`configs/example.yaml`](configs/example.yaml), which documents every
request field inline.

## Input modes

- **Model-in**: you hand over checkpoints; the library probes them itself (unlocks the model-mutating relearning attack).
- **Outputs-in**: you run your own model and submit per-example outputs in the documented payload format; the library scores outputs and never loads the model. Metrics that need a live model skip with `requires_model_in`.

## Checkpoint roles

| Role | Required | Notes |
|---|---|---|
| `unlearned` | yes (model-in) | the subject under test |
| `original` | optional | **identity-bound — user only.** Deltas and pre-attack baselines need the *actual* parent model; it cannot be derived |
| `gold` | optional | retrained-from-scratch reference; expensive but **derivable** (`hp.references.gold="build"`), method-independent, cached and amortized across methods |

Partial availability degrades monotonically: missing roles produce skips that name their unlock, never crashes or silent omissions.

## Guarantees (G1–G10)

- **G1 Determinism** — identical `(request, seed, library_version)` give identical reports up to documented float tolerance.
- **G2 Single-seed discipline** — one seed per request; all randomness via named substreams in `core/seeding.py` (lint-enforced).
- **G3 Provenance completeness** — every number traces to a checkpoint SHA-256, dataset fingerprint, and library/code version; incomplete provenance refuses to serialize.
- **G4 No silent skips** — every uncomputable metric appears in `report.skipped` with a machine-readable code; runtime failures fail soft.
- **G5 Uncertainty everywhere** — every `MetricResult` carries a per-example bootstrap CI; cross-seed dispersion lives only in the aggregator.
- **G6 Hidden-preprocessing disclosure** — every internal preprocessing step is fingerprinted in `provenance.preprocessing`.
- **G7 Cache correctness** — content-addressed caches; a hit is behaviorally identical to recomputation and recorded in provenance.
- **G8 Input-mode equivalence** — outputs-in scores equal model-in scores on the same underlying outputs.
- **G9 Frozen protocol** — versioned hyperparameter defaults; every override is stamped into the report.
- **G10 Regression gate** — each release reproduces the shipped baseline matrix within ±1 percentage point per (method, mode, metric).

## Report anatomy

`EvalReport` (schema 1.0): `task / mode / seed / input_mode`; `metrics[category][name] -> MetricResult(value, ci, n, components, cost_s, peak_mem_mb, cache_hit)`; `skipped[name] -> SkipInfo(code, message)`; `warnings`; `hyperparams` (defaults + stamped overrides); `provenance`. Conventions: accuracies on the 0–100 scale; MIA accuracies on 0–1. `report.scores()` gives the lossy float view; the structured dict is the artifact of record. Canonical JSON via `to_json()`, human review via `to_markdown()`. CSV exists **only** in the aggregator, which always emits per-seed rows, cross-seed std, and the checkpoint-hash column.

## Cache layers

Content-addressed under `.trail_cache`: **L1** forward-pass outputs `hash(ckpt, split_fingerprint, seed)` · **L2** gold model `hash(dataset, mode, seed)` · **L3** shadow ensemble stats `hash(dataset, seed, n_shadow)`. Preprocessing fingerprints participate in split fingerprints, so changing any hidden step auto-invalidates L1.

## Non-goals

No unlearning-method implementations (methods arrive as checkpoints; reference training is the one exception). No hosted service; single-node, single-GPU; models must fit on one device. Shadow budget fixed at 8.

## v0.1 scope

Classification (CIFAR-10/100 with ResNet-18) is the implemented modality and
the scope of the contribution. The LLM adapter ships as a payload
specification raising `NotImplementedError`; activating it is future work, and
the intended path is backend adoption rather than a second evaluation stack.
Generative modalities are out of scope for this release.

## Citation

If you use TRAIL, please cite the accompanying paper:

```bibtex
@inproceedings{trail2026,
  title     = {{TRAIL}: Navigating the Jungle of Machine Unlearning Evaluation},
  author    = {Khodadadi Hosseinabadi, Sepehr and Iakovleva, Ekaterina and
               Tartaglione, Enzo and Pastore, Vito Paolo},
  booktitle = {European Conference on Computer Vision (ECCV) Workshops --
               Unlearning and Model Editing (U\&Me)},
  year      = {2026},
}
```

## License

MIT — see [`LICENSE`](LICENSE). Portions of the membership-inference code are
adapted from [OPTML-Group/Unlearn-Sparse](https://github.com/OPTML-Group/Unlearn-Sparse)
(MIT); its notice is reproduced at the end of `LICENSE`.
