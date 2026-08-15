"""Cross-report aggregation — the ONLY CSV writer in the package.

A table implies a comparison, and comparisons must carry multi-seed
uncertainty (G5), so single-report CSV export deliberately does not exist:
this module merges reports across seeds/methods, always emitting per-seed
rows, per-group mean and std rows, and the checkpoint-hash column (G3).
"""
from __future__ import annotations

import csv
import itertools
import logging
import math
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from trail.core.report import EvalReport

logger = logging.getLogger("trail.aggregate")

#: Identity columns, in the fixed order they lead every CSV.
IDENTITY_COLUMNS: tuple[str, ...] = ("label", "task", "mode", "seed",
                                     "checkpoint_sha")


def load_reports(paths: Iterable[str | Path]) -> list["EvalReport"]:
    """Load serialized reports via ``EvalReport.from_json``.

    Args:
        paths: report JSON files (e.g. ``runs/*.json``).

    Returns:
        Reports in input order.
    """
    from trail.core.report import EvalReport  # lazy: keep module light
    return [EvalReport.from_json(str(p)) for p in paths]


def _unlearned_sha(report: "EvalReport") -> str:
    """Checkpoint SHA-256 of the ``unlearned`` role, or ``""`` (outputs-in)."""
    prov = report.provenance
    if isinstance(prov, Mapping):
        sha_map = prov.get("checkpoint_sha256") or {}
    else:
        sha_map = getattr(prov, "checkpoint_sha256", None) or {}
    return str(sha_map.get("unlearned") or "")


def flatten(report: "EvalReport", label: str) -> dict[str, object]:
    """Flatten one report into a single CSV row dict.

    Columns: the identity columns, then ``f"{category}.{name}"`` for every
    metric value plus ``.ci_lo`` / ``.ci_hi`` companions for its bootstrap CI.

    Args:
        report: a deserialized :class:`EvalReport`.
        label: human-readable method label for the ``label`` column.
    """
    row: dict[str, object] = {
        "label": label,
        "task": report.task,
        "mode": report.mode,
        "seed": report.seed,
        "checkpoint_sha": _unlearned_sha(report),
    }
    for category in sorted(report.metrics):
        for name in sorted(report.metrics[category]):
            res = report.metrics[category][name]
            col = f"{category}.{name}"
            row[col] = float(res.value)
            ci = getattr(res, "ci", None) or (float("nan"), float("nan"))
            row[f"{col}.ci_lo"] = float(ci[0])
            row[f"{col}.ci_hi"] = float(ci[1])
    return row


def _label_for(sha: str, labels: Mapping[str, str] | None) -> str:
    """Resolve a label from sha-prefix mapping; default is ``sha[:8]``."""
    if labels:
        for prefix, label in labels.items():
            if prefix and sha.startswith(prefix):
                return label
    return sha[:8]


def _numeric(value: object) -> float | None:
    """Return a finite float, else None (blanks/strings/NaN excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _group_row(group_rows: Sequence[dict[str, object]],
               group_by: Sequence[str], key: tuple[str, ...],
               metric_cols: Sequence[str], kind: str) -> dict[str, object]:
    """Build one ``mean`` or ``std`` summary row for a group."""
    row: dict[str, object] = {c: "" for c in IDENTITY_COLUMNS}
    for col, val in zip(group_by, key):
        if col in row:
            row[col] = val
    row["seed"] = kind  # literal "mean" / "std" in the seed column
    # Non-grouped identity columns: keep when unique across the group.
    for col in ("task", "checkpoint_sha"):
        if col not in group_by:
            vals = {str(r.get(col, "")) for r in group_rows}
            row[col] = vals.pop() if len(vals) == 1 else ""
    for col in metric_cols:
        vals = [v for r in group_rows
                if (v := _numeric(r.get(col))) is not None]
        if not vals:
            row[col] = ""
        elif kind == "mean":
            row[col] = sum(vals) / len(vals)
        else:  # std, ddof=1; a single seed has no dispersion -> 0.0
            row[col] = 0.0 if len(vals) == 1 else statistics.stdev(vals)
    return row


#: Columns for the tidy/long export, in fixed order.
LONG_COLUMNS: tuple[str, ...] = ("label", "task", "mode", "seed",
                                 "checkpoint_sha", "category", "metric",
                                 "value", "ci_lo", "ci_hi", "n")

#: Expected seed set for a complete multi-seed group (project convention).
EXPECTED_SEEDS: tuple[int, ...] = (2, 3, 42)


def _find_metric(report: "EvalReport", name: str):
    """Return the MetricResult named ``name`` (searching all categories), or None."""
    for results in report.metrics.values():
        if name in results:
            return results[name]
    return None


def _metric_scalar(report: "EvalReport", name: str,
                   component: str | None = None) -> float:
    """Value of metric ``name`` (or its ``component``); ``nan`` if absent."""
    res = _find_metric(report, name)
    if res is None:
        return float("nan")
    if component is None:
        return float(res.value)
    comps = getattr(res, "components", None) or {}
    return float(comps.get(component, float("nan")))


# ---------------------------------------------------------------------------
# F7 — composite-metric convention (Phase-0 scaffold).
#
# CONVENTION (process gate, no code can enforce it):
#   1. "No metric without a registered M-ID." A composite is a metric; before
#      one is registered here it MUST reserve a row in
#      registry rule for primitive metrics (core/registry.register_metric).
#   2. Adding a new metric *category* is a COUPLED edit: core/registry.py's
#      ``Category`` Literal AND the ``CATEGORIES`` tuple must change together
#      (half-applying hard-fails import) plus the metrics/__init__ import smoke.
#
# Composites live HERE — in the aggregator, downstream of EvalReport.to_json /
# Provenance.validate_complete (the G3 chokepoint) — and read FINISHED metric
# values via ``_metric_scalar`` / ``_find_metric``. They are therefore a
# read-only projection of an already-provenance-stamped report; a composite can
# never be injected into a report before serialization, so it cannot bypass the
# provenance gate (G3) and never explodes report JSON.
#
# COMPOSITE_SPECS is intentionally EMPTY in Phase 0. The actual composites
# (defender_utility, AGL, H-LR, AUS, UQS, ...) land in Tier 1 (Phase 3) under
# their firewall constraints (UQS uniform-fallback + loud warning when the
# oracle is absent; AGL sign-safety; composites carry no CI — the documented G5
# carve-out). No composites are defined yet.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier-1 composite definitions (M41-M45). Point estimates, NO CI (documented
# G5 carve-out). Per-report composites read FINISHED metric values via
# _metric_scalar / _find_metric (downstream of provenance). AGR is intentionally
# DEFERRED until cka_to_gold (M33) ships AND a pinned [0,1] normalization of
# activation_distance is defined.
# ---------------------------------------------------------------------------

#: the SIGNED M21 (sum_delta_to_gold) split components AGL multiplies over.
_AGL_SPLITS = ("d_ua", "d_ra_train", "d_ra_test", "d_ta")


def _composite_agl(report: "EvalReport") -> float:
    """M41 AGL: ``∏_split clip(1-|Δ_split|/100, 0, 1)`` over the SIGNED M21
    components ONLY (never M7 — that double-counts sign/space). nan without
    gold/M21."""
    res = _find_metric(report, "sum_delta_to_gold")
    comps = (getattr(res, "components", None) or {}) if res is not None else {}
    if not all(s in comps for s in _AGL_SPLITS):
        return float("nan")
    prod = 1.0
    for s in _AGL_SPLITS:
        prod *= min(max(1.0 - abs(float(comps[s])) / 100.0, 0.0), 1.0)
    return float(prod)


def _harmonic(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or (a + b) <= 0:
        return float("nan")
    return float(2.0 * a * b / (a + b))


def _composite_hlr(report: "EvalReport") -> float:
    """M42 H-LR: harmonic mean of forget-quality (UA) and utility (TA)."""
    return _harmonic(_metric_scalar(report, "ua"), _metric_scalar(report, "ta"))


def _composite_aus(report: "EvalReport") -> float:
    """M43 AUS: utility-penalized forgetting = ``UA·TA/100`` (reference-free).
    Provisional formula."""
    ua = _metric_scalar(report, "ua")
    ta = _metric_scalar(report, "ta")
    if not (math.isfinite(ua) and math.isfinite(ta)):
        return float("nan")
    return float(ua * ta / 100.0)


def _composite_defender_utility(report: "EvalReport") -> float:
    """M44 defender_utility: ``TA − 100·MIA_advantage`` (reference-free; no gold;
    survives ImageNet scale). nan if TA or the MIA advantage is absent."""
    ta = _metric_scalar(report, "ta")
    adv = _metric_scalar(report, "mia_threshold_population", "advantage")
    if not (math.isfinite(ta) and math.isfinite(adv)):
        return float("nan")
    return float(ta - 100.0 * adv)


#: name -> callable(EvalReport) -> float.
COMPOSITE_SPECS: dict[str, Callable[["EvalReport"], float]] = {
    "AGL": _composite_agl,
    "H-LR": _composite_hlr,
    "AUS": _composite_aus,
    "defender_utility": _composite_defender_utility,
}


def compute_composites(
        report: "EvalReport",
        specs: "Mapping[str, Callable[[EvalReport], float]] | None" = None,
        ) -> dict[str, float]:
    """Compute composite scores for a finished report (F7 scaffold).

    Each spec in ``specs`` (default :data:`COMPOSITE_SPECS`) is a callable that
    reads the report's metric values — via :func:`_metric_scalar` /
    :func:`_find_metric` — and returns one float. This runs downstream of
    ``EvalReport.to_json`` (the report is already provenance-complete), so it is
    a pure read: it never mutates ``report`` and never injects a metric ahead of
    the G3 provenance gate. Returns ``{name: value}``; ``{}`` when no composites
    are registered (the Phase-0 state).
    """
    specs = COMPOSITE_SPECS if specs is None else specs
    return {name: float(fn(report)) for name, fn in specs.items()}


def _as_reports(reports: "Sequence") -> "list[EvalReport]":
    """Accept already-loaded EvalReports or report paths."""
    items = list(reports)
    if items and isinstance(items[0], (str, Path)):
        return load_reports(items)
    return items


#: UQS sub-scores: one reference-free (defender_utility) + one gold-anchored (AGL).
_UQS_SUBSCORES: tuple[str, ...] = ("defender_utility", "AGL")


def compute_uqs(reports: "Sequence", *,
                subscores: "Sequence[str]" = _UQS_SUBSCORES,
                labels: Mapping[str, str] | None = None) -> dict:
    """M45 UQS — cohort-weighted unlearning-quality score per method.

    Sub-score weights are derived from the cohort's gold-anchored M21 (each
    sub-score weighted by its across-cohort discriminativeness) ONLY when **≥3
    methods carry a finite M21**; otherwise the weights are forced **uniform**
    with a LOUD warning, so an absent/scale-skipped oracle can never leak into
    the weighting (the firewall). The ``weight_source`` is stamped. Composites
    carry no CI (G5). Returns ``{uqs, weight_source, weights, n_oracle}``.
    """
    reps = _as_reports(reports)
    rows = [compute_composites(rep) for rep in reps]
    n_oracle = sum(1 for rep in reps
                   if math.isfinite(_metric_scalar(rep, "sum_delta_to_gold")))
    subscores = tuple(subscores)
    if n_oracle >= 3:
        stds: dict[str, float] = {}
        for s in subscores:
            vals = [r[s] for r in rows if math.isfinite(r.get(s, float("nan")))]
            stds[s] = float(np.std(vals)) if len(vals) >= 2 else 0.0
        total = sum(stds.values())
        if total > 0:
            weights = {s: stds[s] / total for s in subscores}
            source = "cohort_m21"
        else:
            weights = {s: 1.0 / len(subscores) for s in subscores}
            source = "uniform_degenerate"
    else:
        weights = {s: 1.0 / len(subscores) for s in subscores}
        source = "uniform_no_oracle"
        logger.warning(
            "UQS: gold-anchored M21 finite for only %d (<3) methods; forcing "
            "UNIFORM sub-score weights to avoid oracle leakage", n_oracle)
    lbls = [_label_for(_unlearned_sha(rep), labels) for rep in reps]
    uqs: dict[str, float] = {}
    for lbl, r in zip(lbls, rows):
        finite = [(s, r[s]) for s in subscores
                  if math.isfinite(r.get(s, float("nan")))]
        uqs[lbl] = (float(sum(weights[s] * v for s, v in finite)) if finite
                    else float("nan"))
    return {"uqs": uqs, "weight_source": source, "weights": weights,
            "n_oracle": n_oracle}


#: pinned scipy kendalltau tie-handling variant (determinism across versions).
_RELIABILITY_TIE = "b"


def reliability_table(reports: "Sequence", *, metrics: "Sequence[str]",
                      labels: Mapping[str, str] | None = None) -> list[dict]:
    """Kendall-τ + Spearman-ρ rank correlation between each pair of ``metrics``
    across the method cohort. The cohort is canonically label-sorted (so input
    order is irrelevant) and scipy's tie handling is pinned -> deterministic.
    A pair with <3 jointly-finite methods yields nan."""
    from scipy import stats

    reps = _as_reports(reports)
    lbls = [_label_for(_unlearned_sha(rep), labels) for rep in reps]
    order = sorted(range(len(reps)), key=lambda i: lbls[i])  # canonical sort
    cols = {m: np.array([_metric_scalar(reps[i], m) for i in order],
                        dtype=np.float64) for m in metrics}
    rows: list[dict] = []
    for a, b in itertools.combinations(metrics, 2):
        x, y = cols[a], cols[b]
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) >= 3:
            tau, _ = stats.kendalltau(x[mask], y[mask], variant=_RELIABILITY_TIE)
            rho, _ = stats.spearmanr(x[mask], y[mask])
            tau, rho = float(tau), float(rho)
        else:
            tau = rho = float("nan")
        rows.append({"metric_a": a, "metric_b": b, "kendall_tau": tau,
                     "spearman_rho": rho, "n": int(mask.sum())})
    return rows


# ---------------------------------------------------------------------------
# C4 — forget<->utility trade-off table (one point per method/seed).
# ---------------------------------------------------------------------------

TRADE_OFF_COLUMNS: tuple[str, ...] = (
    "label", "mode", "seed", "checkpoint_sha",
    "forget_axis", "utility_axis", "privacy_advantage")


def trade_off_table(report_paths: Sequence[str | Path], out_csv: str | Path,
                    *, labels: Mapping[str, str] | None = None,
                    forget_metric: str = "ua", utility_metric: str = "ta",
                    privacy_metric: str = "mia_threshold_population",
                    ) -> Path:
    """Assemble the forget<->utility trade-off points (C4).

    Each row is one (method, seed) operating point — ``forget_axis`` (default
    UA, higher = more forgotten), ``utility_axis`` (default mode-aware test
    accuracy TA), and ``privacy_advantage`` (the MIA advantage, 0 = ideal).
    The output is a tidy table ready to plot as a forget<->utility / privacy
    Pareto scatter; the library itself draws nothing (no matplotlib dep).
    """
    reports = load_reports(report_paths)
    if not reports:
        raise ValueError("trade_off_table: no reports given")
    rows: list[dict[str, object]] = []
    for rep in reports:
        sha = _unlearned_sha(rep)
        rows.append({
            "label": _label_for(sha, labels), "mode": rep.mode,
            "seed": rep.seed, "checkpoint_sha": sha,
            "forget_axis": _metric_scalar(rep, forget_metric),
            "utility_axis": _metric_scalar(rep, utility_metric),
            "privacy_advantage": _metric_scalar(rep, privacy_metric, "advantage"),
        })
    out_path = Path(out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRADE_OFF_COLUMNS), restval="")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("trade-off table: %d points -> %s", len(rows), out_path)
    return out_path


# ---------------------------------------------------------------------------
# C6 — significance test across method groups (paired over shared seeds).
# ---------------------------------------------------------------------------

COMPARE_COLUMNS: tuple[str, ...] = (
    "label", "baseline", "metric", "n_pairs",
    "mean_delta", "t_stat", "p_value")


def compare_reports(report_paths: Sequence[str | Path], *, baseline: str,
                    metric: str, out_csv: str | Path | None = None,
                    labels: Mapping[str, str] | None = None,
                    ) -> list[dict[str, object]]:
    """Paired significance test of each method vs a baseline label (C6).

    Groups reports by label; for every non-baseline label, pairs the chosen
    ``metric`` over the seeds shared with the baseline and runs a paired
    t-test (``scipy.stats.ttest_rel``). Small-n caveat: with the usual 3 seeds
    the test has low power — report it alongside mean+-std, not as sole
    evidence. Returns one record per compared label (also written as CSV when
    ``out_csv`` is given).
    """
    from scipy import stats  # ships with scikit-learn
    reports = load_reports(report_paths)
    by_label: dict[str, dict[int, float]] = {}
    for rep in reports:
        lbl = _label_for(_unlearned_sha(rep), labels)
        by_label.setdefault(lbl, {})[int(rep.seed)] = _metric_scalar(rep, metric)
    if baseline not in by_label:
        raise ValueError(f"compare_reports: baseline label {baseline!r} not "
                         f"found among {sorted(by_label)}")
    base = by_label[baseline]
    rows: list[dict[str, object]] = []
    for lbl, seed_vals in sorted(by_label.items()):
        if lbl == baseline:
            continue
        seeds = sorted(set(seed_vals) & set(base))
        a = np.array([seed_vals[s] for s in seeds], dtype=np.float64)
        b = np.array([base[s] for s in seeds], dtype=np.float64)
        mean_delta = float(np.mean(a - b)) if seeds else float("nan")
        if len(seeds) >= 2 and np.any(a - b != 0.0):
            t_stat, p_value = stats.ttest_rel(a, b)
            t_stat, p_value = float(t_stat), float(p_value)
        else:  # <2 pairs or zero variance -> test undefined
            t_stat, p_value = float("nan"), float("nan")
        rows.append({"label": lbl, "baseline": baseline, "metric": metric,
                     "n_pairs": len(seeds), "mean_delta": mean_delta,
                     "t_stat": t_stat, "p_value": p_value})
    if out_csv is not None:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COMPARE_COLUMNS),
                                    restval="")
            writer.writeheader()
            writer.writerows(rows)
        logger.info("compare vs %r on %r: %d rows -> %s",
                    baseline, metric, len(rows), out_csv)
    return rows


def _warn_seed_coverage(groups: "dict[tuple[str, ...], list[dict[str, object]]]",
                        expected_seeds: "tuple[int, ...]") -> None:
    """Log a warning for any group missing an expected seed or with <3 seeds
    (C6: the multi-seed convention is enforced at aggregation — a single run is
    legitimately one seed, so this is the only place coverage can be checked)."""
    want = set(expected_seeds)
    for key, rows in groups.items():
        seeds: set[int] = set()
        for r in rows:
            try:
                seeds.add(int(r.get("seed")))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass  # mean/std rows carry non-int seed labels
        missing = want - seeds
        if missing or len(seeds) < 3:
            logger.warning("seed coverage for group %s: have %s, expected >= %s "
                           "(missing %s) — multi-seed reporting incomplete",
                           key, sorted(seeds), sorted(want), sorted(missing))


def aggregate_reports_long(report_paths: Sequence[str | Path],
                           out_csv: str | Path,
                           *,
                           labels: Mapping[str, str] | None = None,
                           ) -> Path:
    """Merge reports into a tidy/long CSV: ONE ROW PER METRIC.

    The wide :func:`aggregate_reports` pivots metrics into columns; this
    melts them so each row is a single ``(report, category, metric)`` record
    — the tidy-data shape preferred for downstream plotting/stats. Columns:
    ``label, task, mode, seed, checkpoint_sha, category, metric, value,
    ci_lo, ci_hi, n``. No mean/std summary rows (long format is grouped at
    analysis time); identity is carried on every row.

    Args:
        report_paths: report JSON files to merge.
        labels: optional ``{sha256-prefix: label}`` mapping (see
            :func:`aggregate_reports`).
        out_csv: destination CSV path.

    Returns:
        The written CSV path.
    """
    reports = load_reports(report_paths)
    if not reports:
        raise ValueError("aggregate_reports_long: no reports given")

    out_rows: list[dict[str, object]] = []
    for rep in reports:
        label = _label_for(_unlearned_sha(rep), labels)
        sha = _unlearned_sha(rep)
        for category in sorted(rep.metrics):
            for name in sorted(rep.metrics[category]):
                res = rep.metrics[category][name]
                ci = getattr(res, "ci", None) or (float("nan"), float("nan"))
                out_rows.append({
                    "label": label, "task": rep.task, "mode": rep.mode,
                    "seed": rep.seed, "checkpoint_sha": sha,
                    "category": category, "metric": name,
                    "value": float(res.value),
                    "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                    "n": int(getattr(res, "n", 0) or 0),
                })

    out_path = Path(out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LONG_COLUMNS), restval="")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    logger.info("aggregated %d reports into %d metric rows (long) -> %s",
                len(reports), len(out_rows), out_path)
    return out_path


def aggregate_reports(report_paths: Sequence[str | Path],
                      out_csv: str | Path,
                      *,
                      labels: Mapping[str, str] | None = None,
                      group_by: Sequence[str] = ("label", "mode"),
                      expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
                      ) -> Path:
    """Merge reports into one CSV with per-seed rows and group mean/std rows.

    Args:
        report_paths: report JSON files to merge.
        labels: optional ``{sha256-prefix: label}`` mapping; a report whose
            unlearned-checkpoint hash starts with a key gets that label,
            otherwise the label defaults to ``sha[:8]``.
        group_by: identity columns defining the mean/std groups.
        expected_seeds: the multi-seed convention; a group missing any of these
            (or with <3 seeds) is warned about (C6). Pass ``()`` to disable.
        out_csv: destination CSV path.

    Returns:
        The written CSV path. Columns are the union across reports, identity
        columns first, metric columns sorted; missing cells are blank.
    """
    reports = load_reports(report_paths)
    if not reports:
        raise ValueError("aggregate_reports: no reports given")
    rows = [flatten(rep, _label_for(_unlearned_sha(rep), labels))
            for rep in reports]

    metric_cols = sorted({c for r in rows for c in r
                          if c not in IDENTITY_COLUMNS})
    columns = list(IDENTITY_COLUMNS) + metric_cols

    groups: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(c, "")) for c in group_by)
        groups.setdefault(key, []).append(row)

    if expected_seeds:
        _warn_seed_coverage(groups, expected_seeds)

    out_rows: list[dict[str, object]] = []
    for key in sorted(groups):
        per_seed = sorted(groups[key], key=lambda r: str(r.get("seed", "")))
        out_rows.extend(per_seed)
        out_rows.append(_group_row(per_seed, group_by, key, metric_cols,
                                   "mean"))
        out_rows.append(_group_row(per_seed, group_by, key, metric_cols,
                                   "std"))

    out_path = Path(out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    logger.info("aggregated %d reports into %d rows -> %s",
                len(reports), len(out_rows), out_path)
    return out_path
