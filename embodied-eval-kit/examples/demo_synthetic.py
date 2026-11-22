"""Zero-dependency demo: why the pair metric decides, and why the gates decide.

Run it with no data and no install::

    python examples/demo_synthetic.py

Two policies are scored on the same 12 scenes.  Policy A really does follow the
instruction; policy B ignores it and always drops the box on the right.  Watch what each
headline number says about B -- and how the clean-success gates reject a rollout that a
naive "did it lift?" criterion would happily count.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeval import (  # noqa: E402
    CleanCfg,
    Episode,
    Protocol,
    Rollout,
    analyze,
    judge_episode,
    two_proportion_diff,
)

N_SCENES = 12
LEFT, RIGHT = "put it in the left box", "put it in the right box"
EXPECTED = {LEFT: "box_a", RIGHT: "box_b"}
PROTOCOL = Protocol(instructions=(LEFT, RIGHT), expected_box=EXPECTED)


def build(always_right: bool) -> list[Rollout]:
    """Replay every scene under both instructions."""
    rows: list[Rollout] = []
    for scene in range(N_SCENES):
        for index, instruction in enumerate(PROTOCOL.instructions):
            if always_right:
                box = EXPECTED[RIGHT]
            elif instruction == RIGHT and scene < 2:
                box = None  # a real policy also drops two right-hand scenes
            else:
                box = EXPECTED[instruction]
            rows.append(Rollout(episode_id=scene, seed=0, instruction=index, landed_box=box))
    return rows


def main() -> int:
    for label, always_right in (("policy A (follows the instruction)", False),
                                ("policy B (ignores it, always right)", True)):
        report = analyze(build(always_right), PROTOCOL)
        print(f"\n=== {label} ===")
        print(f"  follow_acc          {report.follow_acc:.1%}  "
              f"({report.n_followed}/{report.n_landed} rollouts that landed)")
        print(f"  follow_acc (ITT)    {report.follow_acc_itt:.1%}  "
              f"({report.n_followed}/{report.n_rollouts} rollouts)")
        print(f"  pairs both followed {report.pairs_both_follow}/{report.complete_pairs}")
        print(f"  verdict             {report.verdict.label}")
        print(f"  chance baselines    {report.chance_baselines}")

    # A 12-trial A/B between the two policies, the way a real robot comparison is run.
    a, b = analyze(build(False), PROTOCOL), analyze(build(True), PROTOCOL)
    print(f"\nA vs B (ITT): {two_proportion_diff(a.n_followed, a.n_rollouts, b.n_followed, b.n_rollouts)}")

    # Now the gates: a rollout that digs into the box before lifting is not a success.
    honest = Episode(lift_mm=[0, 20, 60, 120, 130, 130, 130],
                     tilt_deg=[0, 1, 1, 2, 2, 2, 2],
                     penetration_mm=[0, 0.1, 0.2, 0.3, 0.3, 0.3, 0.3])
    cheating = Episode(lift_mm=[0, 5, 70, 130, 135, 130, 120],
                       tilt_deg=[0, 1, 2, 3, 3, 3, 3],
                       penetration_mm=[0, 1.2, 3.4, 3.1, 2.9, 2.8, 2.6])
    for label, episode in (("honest", honest), ("digs in first", cheating)):
        verdict = judge_episode(episode, CleanCfg())
        print(f"\n{label}: ok={verdict['ok']} lift={verdict['max_lift_mm']}mm "
              f"gates_failed={verdict['failed_gates']} cheat={verdict['cheat_flags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())