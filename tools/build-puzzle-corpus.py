#!/usr/bin/env python3
"""Rebuild `tests/data/puzzles.txt` from the labelled PlayTak puzzle database.

The database (`~/MEGA/PTN/puzzles/puzzles.db`, table `puzzles`) holds ~30k
real-game positions carrying an odd `tinue_length` label. Those labels are
**not** trustworthy on their own: rows labelled `tinue_length = 3` have been
confirmed as mates in 5, 7 and 9. This script therefore treats the DB purely
as a source of real, contested positions and re-derives every ply count with
the solver, discarding anything it cannot confirm.

That makes the output a *regression* corpus, not an independent oracle — it
pins current behaviour so a future change that alters a verdict or shortens a
mate shows up as a failure. Independent correctness comes from the Topaz
differential (`tinue-differential.py`), which is a separate engine.

Usage::

    tools/build-puzzle-corpus.py                      # default tiers
    tools/build-puzzle-corpus.py --per-tier 400 --out tests/data/puzzles.txt
    tools/build-puzzle-corpus.py --dry-run            # report, write nothing

Requires `cargo build --release --bin tinue`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sqlite3
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTAKS = REPO / "target" / "release" / "tinue"
DB_DEFAULT = pathlib.Path.home() / "MEGA" / "PTN" / "puzzles" / "puzzles.db"
OUT_DEFAULT = REPO / "tests" / "data" / "puzzles.txt"

# Tiers mirror how the integration test consumes the file: the fast tier runs
# on every `cargo test`, the rest sit behind `#[ignore]`. Keeping deep mates
# out of the default run is what stops the suite from taking minutes.
TIERS = [(3, "fast"), (5, "fast"), (7, "slow"), (9, "deep"), (11, "deep")]


def fetch(db: pathlib.Path, length: int, limit: int) -> list[tuple[int, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [
            (int(s), t)
            for s, t in con.execute(
                """
                SELECT g.size, p.tps
                FROM puzzles p JOIN games g ON g.id = p.game_id
                WHERE p.tinue_length = ?
                ORDER BY p.tps
                LIMIT ?
                """,
                (length, limit),
            )
        ]
    finally:
        con.close()


def solve_batch(cases: list[tuple[int, str]], max_plies: int, max_nodes: int,
                tt_bits: int, scope: str = "full") -> list[dict]:
    """One `--batch` invocation for the whole list — a process spawn per
    position dominates everything else at this scale."""
    stdin = "".join(f"{sz} {tps}\n" for sz, tps in cases)
    out = subprocess.run(
        [str(SYNTAKS), "--batch", "--scope", scope,
         "--max-plies", str(max_plies), "--max-nodes", str(max_nodes),
         "--tt-bits", str(tt_bits)],
        input=stdin, capture_output=True, text=True, check=True,
    )
    return [json.loads(line) for line in out.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=pathlib.Path, default=DB_DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DEFAULT)
    ap.add_argument("--per-tier", type=int, default=300,
                    help="candidates to pull per labelled length (default 300)")
    ap.add_argument("--max-nodes", type=int, default=3_000_000,
                    help="per-depth node budget while verifying (default 3M)")
    ap.add_argument("--tt-bits", type=int, default=22)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SYNTAKS.exists():
        print(f"missing {SYNTAKS}; run: cargo build --release --bin tinue", file=sys.stderr)
        return 2
    if not args.db.exists():
        print(f"puzzle DB not found at {args.db}", file=sys.stderr)
        return 2

    verified: list[tuple[int, int, str, str]] = []  # (size, plies, tps, tier)
    stats: collections.Counter = collections.Counter()

    for length, tier in TIERS:
        cases = fetch(args.db, length, args.per_tier)
        if not cases:
            continue
        # Search a little past the label: the DB understates on some rows, and
        # a mate we confirm at 5 is just as good a fixture as one at 3.
        cap = length + 2
        results = solve_batch(cases, cap, args.max_nodes, args.tt_bits)
        for (size, tps), r in zip(cases, results):
            stats[f"label{length}:{r.get('verdict', 'error')}"] += 1
            if r.get("verdict") != "tinue":
                continue
            plies = r["plies"]
            verified.append((size, plies, tps, tier))
        print(f"  label {length:>2} ({tier:>4}): {len(cases):>4} candidates -> "
              f"{sum(1 for r in results if r.get('verdict') == 'tinue'):>4} verified",
              file=sys.stderr)

    # A position can be pulled under more than one label; keep one copy.
    seen = set()
    unique = []
    for row in verified:
        if row[2] in seen:
            continue
        seen.add(row[2])
        unique.append(row)
    unique.sort(key=lambda r: (r[1], r[0], r[2]))

    by_tier = collections.Counter(r[3] for r in unique)
    by_plies = collections.Counter(r[1] for r in unique)
    print(f"\nverified {len(unique)} unique positions", file=sys.stderr)
    print(f"  by tier : {dict(by_tier)}", file=sys.stderr)
    print(f"  by plies: {dict(sorted(by_plies.items()))}", file=sys.stderr)

    if args.dry_run:
        print("(dry run — nothing written)", file=sys.stderr)
        return 0

    header = f"""# Tinue corpus for the syntaks solver regression tests.
# Format: <size> <verified_plies> <tps>
#
# Source: real PlayTak games, via the labelled puzzle database at
# {args.db}. Regenerate with tools/build-puzzle-corpus.py.
#
# IMPORTANT: the ply counts here are the *solver's* verified shortest mate,
# not the database's `tinue_length` column. That column is unreliable — rows
# labelled 3 have been confirmed as mates in 5, 7 and 9 — so it is used only
# to select candidate positions, never as the expected answer.
#
# This is consequently a REGRESSION corpus: it pins current behaviour so a
# change that loses a tinue or reports a longer mate fails loudly. It is not
# an independent correctness oracle. That role belongs to the Topaz
# differential in tools/tinue-differential.py, which is a separate engine.
#
# {len(unique)} positions: {dict(sorted(by_plies.items()))}
"""
    with args.out.open("w") as f:
        f.write(header)
        for size, plies, tps, _tier in unique:
            f.write(f"{size} {plies} {tps}\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
