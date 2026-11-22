"""Command line entry point: ``python -m eeval ...`` (or the ``eeval`` script).

Three subcommands, one per thing a person actually wants to do at 2 a.m.:

``eeval report rows.json --protocol protocol.json``
    Full counterfactual report for a rollout log (markdown table and/or json).
``eeval judge episodes.jsonl``
    Apply the clean-success gates to per-episode traces and print the failure breakdown.
``eeval selfcheck``
    Re-run the numeric fixtures the test suite asserts on; useful right after installing.

Exit codes follow the receipt states, so a shell script can gate on them::

    0  every stage ok / the pre-registered verdict passed
    1  a warning: the report was produced but the verdict did not pass
    2  a failure: bad input, unusable protocol, or a self-check mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .counterfactual import Protocol, analyze_rows
from .judge import CleanCfg, Episode, judge_many
from .receipt import Receipt, exit_code
from .stats import fisher_exact_2x2, mcnemar_exact, wilson


def _load_rows(path: str) -> list[dict]:
    """Accept a JSON list, a JSON object with ``rows``, or JSON lines."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        if isinstance(data, dict):
            if "rows" in data:
                return data["rows"]
            raise SystemExit(f"{path}: JSON object has no 'rows' key")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _protocol_from_args(args) -> Protocol:
    if args.protocol:
        return Protocol.from_json_file(args.protocol)
    if not (args.tasks and args.boxes):
        raise SystemExit("give --protocol FILE, or both --tasks and --boxes")
    tasks = [item.strip() for item in args.tasks.split(",")]
    boxes = [item.strip() for item in args.boxes.split(",")]
    if len(tasks) != len(boxes):
        raise SystemExit(f"--tasks has {len(tasks)} entries but --boxes has {len(boxes)}")
    return Protocol(instructions=tuple(tasks), expected_box=dict(zip(tasks, boxes)))


def _cmd_report(args) -> int:
    rows = _load_rows(args.rows)
    protocol = _protocol_from_args(args)
    report = analyze_rows(rows, protocol)
    if args.markdown:
        Path(args.markdown).write_text(report.to_markdown() + "\n", encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(report.to_json() + "\n", encoding="utf-8")
    if not args.quiet:
        print(report.to_markdown())
    receipt = Receipt().add(
        "counterfactual",
        "ok" if report.verdict.passed else "warn",
        note=report.verdict.label,
        follow_acc=report.follow_acc,
        itt=report.follow_acc_itt,
        pairs_both_follow=f"{report.pairs_both_follow}/{report.complete_pairs}",
    )
    if args.receipt:
        Path(args.receipt).write_text(receipt.to_json() + "\n", encoding="utf-8")
    return exit_code(receipt)


def _cmd_judge(args) -> int:
    records = _load_rows(args.episodes)
    episodes = [
        Episode(
            lift_mm=record.get("lift_mm", []),
            tilt_deg=record.get("tilt_deg", []),
            penetration_mm=record.get("penetration_mm", []),
            landed_box=record.get("landed_box"),
            episode_id=record.get("episode_id"),
        )
        for record in records
    ]
    cfg = CleanCfg(
        lift_min_mm=args.lift,
        lift_consecutive=args.consecutive,
        tilt_max_deg=args.tilt,
        penetration_max_mm=args.penetration,
        require_release=args.require_release,
    )
    summary = judge_many(episodes, cfg)
    print(f"{summary['n_ok']}/{summary['n']} clean ({summary['success_rate']})")
    for gate, count in summary["by_gate"].items():
        if count:
            print(f"  failed {gate}: {count}")
    for flag, count in summary["by_cheat"].items():
        if count:
            print(f"  cheat pattern '{flag}': {count}")
    if args.verbose:
        for index, verdict in enumerate(summary["per_episode"]):
            print(f"[{index}] {'ok' if verdict['ok'] else 'fail'} {verdict['reasons']}")
    return 0 if summary["n_ok"] == summary["n"] else 1


def _cmd_selfcheck(args) -> int:
    """The handful of numbers that must hold, printed as a three-state receipt."""
    receipt = Receipt()
    checks = [
        ("wilson(40, 40) lower == 0.9124", abs(wilson(40, 40)[0] - 0.9124) < 5e-4),
        ("fisher(1,11,9,3) == 0.0028", abs(fisher_exact_2x2(1, 11, 9, 3) - 0.0028) < 5e-4),
        ("fisher(1,11,8,4) == 0.0094", abs(fisher_exact_2x2(1, 11, 8, 4) - 0.0094) < 5e-4),
        ("mcnemar(8, 1) < 0.05", mcnemar_exact(8, 1) < 0.05),
        ("wilson(0, 40) upper < 0.1", wilson(0, 40)[1] < 0.1),
    ]
    for name, ok in checks:
        receipt.add(name, "ok" if ok else "fail")
    print(receipt.to_markdown())
    return exit_code(receipt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eeval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    report = sub.add_parser("report", help="counterfactual report for a rollout log")
    report.add_argument("rows", help="rows json / jsonl (list, {'rows': [...]}, or jsonl)")
    report.add_argument("--protocol", help="protocol json: {instructions, expected_box}")
    report.add_argument("--tasks", help="comma separated instructions (when no --protocol)")
    report.add_argument("--boxes", help="comma separated expected boxes (when no --protocol)")
    report.add_argument("--json", dest="json_out", help="write the full report here")
    report.add_argument("--markdown", help="write the markdown table here")
    report.add_argument("--receipt", help="write the three-state receipt here")
    report.add_argument("--quiet", action="store_true")
    report.set_defaults(func=_cmd_report)

    judge = sub.add_parser("judge", help="clean-success gates over episode traces")
    judge.add_argument("episodes", help="jsonl with lift_mm / tilt_deg / penetration_mm")
    judge.add_argument("--lift", type=float, default=50.0)
    judge.add_argument("--consecutive", type=int, default=5)
    judge.add_argument("--tilt", type=float, default=5.0)
    judge.add_argument("--penetration", type=float, default=0.5)
    judge.add_argument("--release", dest="require_release", action="store_true")
    judge.add_argument("-v", "--verbose", action="store_true")
    judge.set_defaults(func=_cmd_judge)

    selfcheck = sub.add_parser("selfcheck", help="re-run the numeric fixtures")
    selfcheck.set_defaults(func=_cmd_selfcheck)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())