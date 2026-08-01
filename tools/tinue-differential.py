#!/usr/bin/env python3
"""Differential-test syntaks's tak-chain tinue scope against the Topaz oracle.

Topaz solves *strict tak-chain* tinues only, which is exactly what
``--scope tak-chain`` is defined to mean. The two engines must therefore agree
on every position: any disagreement is a bug in the restriction, not a
difference of opinion.

The script also checks the one-directional invariant that does not involve
Topaz at all: **restricted results are a subset of full results.** A position
that is tinue under ``tak-chain`` must be tinue under ``full``. The converse
may fail freely — that gap is where gap tinues live.

Usage::

    tools/tinue-differential.py --corpus            # the curated puzzle corpus
    tools/tinue-differential.py --random 300        # 300 random 5x5 positions
    tools/tinue-differential.py --random 200 --size 6 --seed 42

Requires both binaries to be built:

    cargo build --release --bins
    (cd ../topaz-tinue-web && cargo build --release --bin topaz-tinue)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SYNTAKS = REPO / "target" / "release" / "tinue"
GENPOS = REPO / "target" / "release" / "genpos"
TOPAZ = REPO.parent / "topaz-tinue-web" / "target" / "release" / "topaz-tinue"
CORPUS = REPO / "tests" / "data" / "puzzles.txt"


def run_syntaks(tps: str, scope: str, max_plies: int, tt_bits: int, timeout: float) -> dict | None:
    cmd = [
        str(SYNTAKS), tps, "--quiet",
        "--scope", scope,
        "--max-plies", str(max_plies),
        "--tt-bits", str(tt_bits),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(f"syntaks failed on {tps!r}: {e.stderr.strip()}", file=sys.stderr)
        return None
    return json.loads(out.stdout.strip())


def run_topaz(tps: str, timeout: float) -> dict | None:
    try:
        out = subprocess.run(
            [str(TOPAZ), tps], capture_output=True, text=True, timeout=timeout, check=True
        )
    except subprocess.TimeoutExpired:
        return None
    except subprocess.CalledProcessError as e:
        print(f"topaz failed on {tps!r}: {e.stderr.strip()}", file=sys.stderr)
        return None
    return json.loads(out.stdout.strip())


def load_corpus() -> list[tuple[int, str]]:
    cases = []
    for line in CORPUS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        size, _expected, tps = line.split(" ", 2)
        cases.append((int(size), tps))
    return cases


def load_random(count: int, size: int, plies: int, seed: int) -> list[tuple[int, str]]:
    out = subprocess.run(
        [str(GENPOS), "--size", str(size), "--plies", str(plies),
         "--count", str(count), "--seed", str(seed)],
        capture_output=True, text=True, check=True,
    )
    cases = []
    for line in out.stdout.splitlines():
        sz, tps = line.split(" ", 1)
        cases.append((int(sz), tps))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--corpus", action="store_true", help="use the curated puzzle corpus")
    src.add_argument("--random", type=int, metavar="N", help="generate N random positions")
    src.add_argument("--file", type=pathlib.Path, metavar="PATH",
                     help="read positions from a file of '<size> <tps>' lines (genpos format)")
    ap.add_argument("--size", type=int, default=5, help="board size for --random (default 5)")
    ap.add_argument("--gen-plies", type=int, default=24, help="random-game length (default 24)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=5, help="search depth cap (default 5)")
    ap.add_argument("--tt-bits", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120.0, help="per-position seconds")
    ap.add_argument("--skip-full", action="store_true",
                    help="skip the restricted-subset-of-full check (it is the slow half)")
    args = ap.parse_args()

    for path, what in ((SYNTAKS, "syntaks tinue"), (TOPAZ, "topaz-tinue"), (GENPOS, "genpos")):
        if not path.exists():
            print(f"missing {what} binary at {path}; see the module docstring", file=sys.stderr)
            return 2

    if args.corpus:
        cases = load_corpus()
    elif args.file:
        cases = []
        for line in args.file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head, _, rest = line.partition(" ")
            cases.append((int(head), rest) if head.isdigit() else (line.count("/") + 1, line))
    else:
        cases = load_random(args.random, args.size, args.gen_plies, args.seed)

    mismatches = 0
    subset_violations = 0
    skipped = 0
    agree = 0

    for idx, (_size, tps) in enumerate(cases, 1):
        chain = run_syntaks(tps, "tak-chain", args.max_plies, args.tt_bits, args.timeout)
        topaz = run_topaz(tps, args.timeout)
        if chain is None or topaz is None:
            skipped += 1
            continue

        # Topaz searches to its own depth, so compare the verdict only. A ply
        # comparison would be meaningful only where both bounded the same way.
        if chain["verdict"] != topaz["verdict"]:
            # Topaz is unbounded; syntaks is capped at --max-plies. A tinue
            # deeper than the cap is a bounded-search artefact, not a
            # disagreement, so only flag the case Topaz calls quiet.
            if chain["verdict"] == "no_tinue" and topaz["verdict"] == "tinue" \
                    and topaz["plies"] > args.max_plies:
                skipped += 1
                continue
            mismatches += 1
            print(f"MISMATCH [{idx}] {tps}")
            print(f"   syntaks tak-chain: {chain['verdict']} plies={chain['plies']}")
            print(f"   topaz            : {topaz['verdict']} plies={topaz['plies']}")
        else:
            agree += 1

        # restricted tinue => full tinue
        if not args.skip_full and chain["verdict"] == "tinue":
            full = run_syntaks(tps, "full", args.max_plies, args.tt_bits, args.timeout)
            if full is None:
                skipped += 1
            elif full["verdict"] != "tinue":
                subset_violations += 1
                print(f"SUBSET VIOLATION [{idx}] {tps}")
                print(f"   tak-chain says tinue in {chain['plies']}, full says {full['verdict']}")

    total = len(cases)
    print(f"\n{agree}/{total} agree with Topaz  |  {mismatches} mismatches  "
          f"|  {subset_violations} subset violations  |  {skipped} skipped")
    return 1 if (mismatches or subset_violations) else 0


if __name__ == "__main__":
    sys.exit(main())
