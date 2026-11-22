"""Rerun the analysis behind this project's published table: real data, no install.

    python examples/demo_real_run.py

The log in ``examples/data/language_pair_76rows.json`` is a real evaluation run of a
language-conditioned ACT policy: 19 held-out scenes, 2 seeds each, both instructions,
76 rollouts.  It is the same data behind the numbers in the project README, so this
script is a worked example and a regression test at the same time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eeval import Protocol, Receipt, analyze_rows, exit_code, verdict  # noqa: E402

DATA = ROOT / "examples" / "data" / "language_pair_76rows.json"


def main() -> int:
    run = json.loads(DATA.read_text(encoding="utf-8"))
    protocol = Protocol.from_dict(run["protocol"])
    report = analyze_rows(run["rows"], protocol)

    print(f"source: {run['source']}\n")
    print(report.to_markdown())

    print("\noutcome breakdown:")
    for name, count in sorted(report.outcome_counts.items()):
        print(f"  {name:<22} {count}")
    print("\nper instruction:")
    for instruction, stats in report.per_instruction.items():
        print(f"  {instruction[:46]:<46} landed {stats['n_landed']:>2}  "
              f"followed {stats['n_followed']:>2}  acc {stats['follow_acc']}")

    receipt = Receipt().add(
        "language_pair_5000",
        "ok" if report.verdict.passed else "warn",
        note=report.verdict.label,
        follow_acc=report.follow_acc,
        itt=report.follow_acc_itt,
        pairs_both_follow=f"{report.pairs_both_follow}/{report.complete_pairs}",
    )
    print("\n" + receipt.to_markdown())
    print(f"\nlegacy dict API says: {verdict(report.as_dict())}")
    return exit_code(receipt)


if __name__ == "__main__":
    sys.exit(main())