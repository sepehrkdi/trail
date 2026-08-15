"""Regression gate: reproduce the shipped baseline matrix.

For every (method, mode, seed) row of the pinned fixture CSV, an
``evaluate()`` call must reproduce the recorded accuracy panel within
±1 percentage point. The fixture ships with the library
(``trail/fixtures/external_baselines_v2.csv``); its checkpoint column holds
file names, resolved against the ``ckpt_root`` directory you supply. Rows
whose checkpoint is absent are reported as ``skipped_missing_ckpt`` rather
than failing the gate. See ``trail/fixtures/README.md`` for fixture
provenance and for where the referenced checkpoints come from.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib import resources
from typing import Collection, Sequence

logger = logging.getLogger("trail.reproduce")

#: CIFAR-10 test-split class sizes used to recombine a full-test accuracy
#: from the derived test partitions in single_class mode (1 forget class
#: x 1000 images, 9 retain classes x 1000 images).
_N_TEST_FORGET = 1000
_N_TEST_RETAIN = 9000
_N_TEST_TOTAL = _N_TEST_FORGET + _N_TEST_RETAIN

#: Metric names requested from evaluate() for every fixture row.
_PANEL: tuple[str, ...] = ("fa_train", "fa_test", "ra_train", "ra_test",
                           "ua", "ta")


@dataclass
class RowResult:
    """Outcome of one fixture row (one method/mode/seed cell)."""

    method: str
    mode: str
    seed: int
    ckpt: str
    status: str  # "ok" | "fail" | "skipped_missing_ckpt"
    deltas: dict[str, float] = field(default_factory=dict)
    max_abs_delta: float = 0.0
    #: Fixture columns that are applicable in the row's mode but for which
    #: the report produced no value (a genuine runtime miss -> row fails).
    missing_columns: list[str] = field(default_factory=list)


@dataclass
class ReproductionResult:
    """Aggregate outcome of the regression gate."""

    rows: list[RowResult]
    n_executed: int
    n_passed: int
    passed: bool
    tolerance_pp: float


def _default_fixture_path() -> str:
    """Path to the pinned fixture CSV shipped inside the package."""
    return str(resources.files("trail")
               .joinpath("fixtures/external_baselines_v2.csv"))


def _report_metric(report, name: str) -> float | None:
    """Look a metric value up across report categories; None if absent."""
    for category in report.metrics:
        if name in report.metrics[category]:
            return float(report.metrics[category][name].value)
    return None


def _expected_values(report, mode: str) -> dict[str, float | None]:
    """Map fixture column names to the report-side comparable values.

    ``ta`` (headline test accuracy) is full-test-based in random mode but
    retain-test-based in single_class mode, so the single_class full-test
    number is recombined from the derived partitions:
    ``(fa_test*1000 + ra_test*9000) / 10000``.

    Columns that are structurally absent in a mode are *omitted* from the
    mapping (skipped, never compared, never a failure). In random mode the
    ``retain_test`` split is None by design, and the fixture follows the
    legacy convention ``retain_test_acc == full_test_acc`` — so that column
    maps to the full-test value (``ta``), and ``forget_test_acc`` (also
    structurally absent) is dropped from the comparison.
    """
    fa_train = _report_metric(report, "fa_train")
    fa_test = _report_metric(report, "fa_test")
    ra_train = _report_metric(report, "ra_train")
    ra_test = _report_metric(report, "ra_test")
    ua = _report_metric(report, "ua")
    ta = _report_metric(report, "ta")

    if mode == "random":
        return {
            "forget_train_acc": fa_train,
            "retain_train_acc": ra_train,
            "full_test_acc": ta,        # random mode: ta is full-test-based
            "retain_test_acc": ta,      # legacy convention: == full-test
            "unlearning_acc": ua,
            "headline_test_acc": ta,
        }

    if fa_test is not None and ra_test is not None:
        full_test = (fa_test * _N_TEST_FORGET
                     + ra_test * _N_TEST_RETAIN) / _N_TEST_TOTAL
    else:
        full_test = None
    return {
        "forget_train_acc": fa_train,
        "retain_train_acc": ra_train,
        "full_test_acc": full_test,
        "retain_test_acc": ra_test,
        "forget_test_acc": fa_test,  # compared only where fixture has a value
        "unlearning_acc": ua,
        "headline_test_acc": ta,
    }


def _row_deltas(report, fixture_row: dict[str, str],
                metrics_subset: Collection[str] | None,
                ) -> tuple[dict[str, float], list[str]]:
    """Per-column ``report - fixture`` deltas (pp), plus missing columns.

    Returns ``(deltas, missing)``. ``deltas`` contains only finite values
    (never NaN/inf). Structurally-absent columns — empty fixture cells, or
    columns dropped from the mode's :func:`_expected_values` mapping — are
    skipped silently. ``missing`` lists columns the fixture populates and
    the mode supports but for which the report has no value: a genuine
    runtime miss the caller must treat as a row failure.
    """
    expected = _expected_values(report, fixture_row["mode"])
    deltas: dict[str, float] = {}
    missing: list[str] = []
    for column, report_value in expected.items():
        cell = (fixture_row.get(column) or "").strip()
        if not cell:  # empty by design (skipped column, never a failure)
            continue
        if metrics_subset is not None and column not in metrics_subset:
            continue
        if report_value is None:
            missing.append(column)
            continue
        deltas[column] = report_value - float(cell)
    return deltas, missing


def reproduce_baseline_matrix(fixture_csv: str | None = None,
                              *,
                              ckpt_root: str | None = None,
                              data_dir: str = "./data",
                              tolerance_pp: float = 1.0,
                              metrics_subset: Collection[str] | None = None,
                              cache_dir: str = ".trail_cache",
                              wandb: bool = False) -> ReproductionResult:
    """Run the G10 regression gate against the pinned baseline fixture.

    Args:
        fixture_csv: fixture CSV path; default is the packaged
            ``external_baselines_v2.csv``.
        ckpt_root: when given, fixture checkpoint paths are remapped by
            basename under this directory; otherwise the fixture's absolute
            paths are used as-is.
        data_dir: CIFAR-10 root passed to the dataset spec.
        tolerance_pp: pass band in percentage points (default ±1 pp).
        metrics_subset: optional subset of fixture column names to compare
            (e.g. ``{"retain_test_acc", "headline_test_acc"}``).
        cache_dir: trail cache directory for the underlying evaluations.
        wandb: enable W&B tracking for the gate runs (off by default —
            CI-oriented entry point).

    Returns:
        A :class:`ReproductionResult`; ``passed`` requires at least one
        executed row and every executed row within tolerance.
    """
    # Lazy imports: the gate touches the full library surface, but importing
    # this module (e.g. from the CLI) must stay cheap.
    import csv

    from trail import evaluate
    from trail.core.request import CacheConfig, Hyperparams, LogConfig
    from trail.data.specs import DatasetSpec

    fixture_path = fixture_csv or _default_fixture_path()
    with open(fixture_path, newline="", encoding="utf-8") as fh:
        fixture_rows = list(csv.DictReader(fh))
    if not fixture_rows:
        raise ValueError(f"empty fixture CSV: {fixture_path}")

    rows: list[RowResult] = []
    for fixture_row in fixture_rows:
        method = fixture_row["method"]
        mode = fixture_row["mode"]
        seed = int(fixture_row["seed"])
        ckpt = fixture_row["ckpt"]
        if ckpt_root is not None:
            ckpt = os.path.join(ckpt_root, os.path.basename(ckpt))

        if not os.path.exists(ckpt):
            logger.warning("skipping %s/%s/seed%d: checkpoint missing at %s",
                           method, mode, seed, ckpt)
            rows.append(RowResult(method=method, mode=mode, seed=seed,
                                  ckpt=ckpt, status="skipped_missing_ckpt"))
            continue

        spec = DatasetSpec(
            mode=mode,
            carveout_seed=int(fixture_row["carveout_seed"]),
            forget_class=0 if mode == "single_class" else None,
            forget_fraction=0.1 if mode == "random" else None,
            split_seed=seed if mode == "random" else 0,
            data_dir=data_dir,
        )
        # Request mode and data mode are two sources of truth; the gate must
        # never evaluate a row's bundle under a different request mode.
        assert spec.mode == mode, (
            f"DatasetSpec.mode={spec.mode!r} disagrees with fixture row "
            f"mode={mode!r} for {method}/seed{seed}")
        logger.info("evaluating %s/%s/seed%d (%s)", method, mode, seed, ckpt)
        try:
            report = evaluate(
                data=spec,
                seed=seed,
                checkpoints={"unlearned": ckpt},
                mode=mode,
                metrics=list(_PANEL),
                hp=Hyperparams(),
                log=LogConfig(wandb=wandb),
                cache=CacheConfig(dir=cache_dir),
            )
            deltas, missing = _row_deltas(report, fixture_row, metrics_subset)
        except Exception:  # noqa: BLE001 — one broken cell must not kill the gate
            logger.exception("evaluation failed for %s/%s/seed%d",
                             method, mode, seed)
            rows.append(RowResult(method=method, mode=mode, seed=seed,
                                  ckpt=ckpt, status="fail",
                                  max_abs_delta=float("inf")))
            continue

        if missing:  # mode-applicable column with no report value: row fails
            logger.error("%s/%s/seed%d: report lacks values for fixture "
                         "column(s) %s", method, mode, seed, missing)
            max_abs = float("inf")
        else:
            max_abs = max((abs(d) for d in deltas.values()), default=0.0)
        status = "ok" if max_abs <= tolerance_pp else "fail"
        rows.append(RowResult(method=method, mode=mode, seed=seed, ckpt=ckpt,
                              status=status, deltas=deltas,
                              max_abs_delta=max_abs, missing_columns=missing))

    n_executed = sum(r.status != "skipped_missing_ckpt" for r in rows)
    n_passed = sum(r.status == "ok" for r in rows)
    passed = n_executed >= 1 and n_passed == n_executed
    if n_executed == 0:
        logger.error("no fixture row was executable (all checkpoints "
                     "missing); gate cannot pass")
    return ReproductionResult(rows=rows, n_executed=n_executed,
                              n_passed=n_passed, passed=passed,
                              tolerance_pp=tolerance_pp)


def format_table(result: ReproductionResult) -> str:
    """Render a :class:`ReproductionResult` as a fixed-width text table."""
    headers: Sequence[str] = ("method", "mode", "seed", "status",
                              "max|d|pp", "worst_column")
    body: list[tuple[str, ...]] = []
    for row in result.rows:
        worst = ""
        if row.missing_columns:
            worst = "missing:" + ",".join(row.missing_columns)
        elif row.deltas:
            worst = max(row.deltas, key=lambda k: abs(row.deltas[k]))
        mad = ("" if row.status == "skipped_missing_ckpt"
               else f"{row.max_abs_delta:.3f}")
        body.append((row.method, row.mode, str(row.seed), row.status,
                     mad, worst))
    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*r) for r in body]
    lines.append("")
    n_skipped = len(result.rows) - result.n_executed
    lines.append(
        f"executed={result.n_executed} passed={result.n_passed} "
        f"skipped={n_skipped} tolerance_pp={result.tolerance_pp:g} "
        f"overall={'PASS' if result.passed else 'FAIL'}")
    return "\n".join(lines)
