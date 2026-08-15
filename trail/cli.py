"""trail command-line interface.

Subcommands mirror the API:

- ``trail eval --config cfg.yaml [-o key=val ...] [--out report.json]``
- ``trail aggregate runs/*.json -o results.csv [--labels labels.yaml]``
- ``trail reproduce baseline-matrix [--ckpt-root DIR] [--tolerance PP]``

Exit codes: 0 success (and gate PASS for ``reproduce``); 1 runtime/gate
failure; 2 usage error (argparse).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Sequence

from trail.io.logging import setup_logging

logger = logging.getLogger("trail.cli")

#: Modules searched (in order) for the YAML request builder.
_REQUEST_BUILDER_MODULES = ("trail.core.request", "trail.core.config",
                            "trail")


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string: ``"auto"`` -> ``"cuda"`` when available,
    else ``"cpu"``; anything else passes through unchanged."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover — torch is a hard dependency
        return "cpu"


def _locate_request_builder() -> Callable:
    """Find ``build_request_from_yaml`` on the documented config surface."""
    for module_name in _REQUEST_BUILDER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        builder = getattr(module, "build_request_from_yaml", None)
        if builder is not None:
            return builder
    raise ImportError(
        "build_request_from_yaml not found on any of: "
        + ", ".join(_REQUEST_BUILDER_MODULES))


def _load_yaml(path: str) -> dict:
    """Load a YAML file into a dict (empty file -> empty dict)."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _cmd_eval(args: argparse.Namespace) -> int:
    """Build the request from YAML + dotted overrides, run, write report."""
    from trail import run  # lazy: full library import

    builder = _locate_request_builder()
    request = builder(args.config, list(args.override))
    report = run(request)
    report.to_json(args.out)
    logger.info("report written: %s", args.out)
    if args.markdown:
        md_path = Path(args.out).with_suffix(".md")
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        logger.info("markdown written: %s", md_path)
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Merge report JSONs into a CSV (wide; --long tidy; --trade-off Pareto)."""
    from trail.aggregate import (aggregate_reports, aggregate_reports_long,
                                   trade_off_table)

    labels = None
    if args.labels:
        raw = _load_yaml(args.labels)
        labels = {str(k): str(v) for k, v in raw.items()}
    if args.long and args.trade_off:
        logger.error("choose at most one of --long / --trade-off")
        return 2
    if args.trade_off:
        trade_off_table(args.reports, args.out, labels=labels)
    elif args.long:
        aggregate_reports_long(args.reports, args.out, labels=labels)
    else:
        aggregate_reports(args.reports, args.out, labels=labels)
    logger.info("aggregate written: %s", args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Paired significance test of each method vs a baseline label (C6)."""
    from trail.aggregate import COMPARE_COLUMNS, compare_reports

    labels = None
    if args.labels:
        raw = _load_yaml(args.labels)
        labels = {str(k): str(v) for k, v in raw.items()}
    rows = compare_reports(args.reports, baseline=args.baseline,
                           metric=args.metric, out_csv=args.out, labels=labels)
    for r in rows:  # echo a compact table to stdout
        print(  # noqa: T201 — CLI output
            " | ".join(f"{c}={r[c]}" for c in COMPARE_COLUMNS))
    return 0




def _cmd_reproduce(args: argparse.Namespace) -> int:
    """Run the G10 regression gate; exit 0 iff the gate passes."""
    from trail.reproduce import format_table, reproduce_baseline_matrix

    result = reproduce_baseline_matrix(
        args.fixture,
        ckpt_root=args.ckpt_root,
        data_dir=args.data_dir,
        tolerance_pp=args.tolerance,
        cache_dir=args.cache_dir,
    )
    print(format_table(result))  # noqa: T201 — the table IS the CLI output
    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True),
            encoding="utf-8")
        logger.info("gate result written: %s", args.out_json)
    return 0 if result.passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trail",
        description="TRAIL: unified machine-unlearning evaluation.")
    parser.add_argument("--log-level", default="INFO",
                        help="stdlib logging level (default: INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="run one evaluation request")
    p_eval.add_argument("--config", required=True,
                        help="YAML request config")
    p_eval.add_argument("-o", "--override", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="dotted config override; repeatable")
    p_eval.add_argument("--out", default="report.json",
                        help="report JSON output path")
    p_eval.add_argument("--markdown", action="store_true",
                        help="also render the report to <out>.md")
    p_eval.set_defaults(func=_cmd_eval)

    p_agg = sub.add_parser("aggregate",
                           help="merge reports into the multi-seed CSV")
    p_agg.add_argument("reports", nargs="+", help="report JSON files")
    p_agg.add_argument("-o", "--out", required=True, help="output CSV path")
    p_agg.add_argument("--labels", default=None,
                       help="YAML mapping checkpoint-sha prefix -> label")
    p_agg.add_argument("--long", action="store_true",
                       help="emit tidy/long format (one row per metric) "
                            "instead of the wide multi-seed table")
    p_agg.add_argument("--trade-off", action="store_true",
                       help="emit the forget<->utility trade-off table (one "
                            "operating point per method/seed)")
    p_agg.set_defaults(func=_cmd_aggregate)

    p_cmp = sub.add_parser("compare",
                           help="paired significance test vs a baseline (C6)")
    p_cmp.add_argument("reports", nargs="+", help="report JSON files")
    p_cmp.add_argument("--baseline", required=True,
                       help="label of the baseline method to test against")
    p_cmp.add_argument("--metric", required=True,
                       help="metric name to compare (e.g. ua, ra_test)")
    p_cmp.add_argument("-o", "--out", default=None, help="output CSV path")
    p_cmp.add_argument("--labels", default=None,
                       help="YAML mapping checkpoint-sha prefix -> label")
    p_cmp.set_defaults(func=_cmd_compare)

    p_rep = sub.add_parser("reproduce", help="run the G10 regression gate")
    p_rep.add_argument("target", choices=["baseline-matrix"],
                       help="what to reproduce")
    p_rep.add_argument("--fixture", default=None,
                       help="fixture CSV (default: packaged baseline matrix)")
    p_rep.add_argument("--ckpt-root", default=None,
                       help="remap fixture checkpoint paths by basename "
                            "under this directory")
    p_rep.add_argument("--data-dir", default="./data",
                       help="dataset root (default: ./data)")
    p_rep.add_argument("--tolerance", type=float, default=1.0,
                       help="pass band in percentage points (default: 1.0)")
    p_rep.add_argument("--out-json", default=None,
                       help="also write the gate result as JSON")
    p_rep.add_argument("--cache-dir", default=".trail_cache",
                       help="content-addressed cache directory")
    p_rep.set_defaults(func=_cmd_reproduce)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        logger.error("interrupted")
        return 130
    except ImportError as exc:
        logger.error("missing component: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("file not found: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — TRAILError + unexpected
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
