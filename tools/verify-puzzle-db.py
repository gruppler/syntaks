#!/usr/bin/env python3
"""Re-derive the tinue labels in the PlayTak puzzle database with syntaks.

## Why

`puzzles.tinue_length` is unreliable, and the failure has a specific shape:
sampling shows labels that are *shorter* than the true shortest mate (250-row
samples: 1.2% wrong at label 3, 15.6% at label 5, 22.4% at label 7, almost all
of them understating). A label shorter than the shortest mate cannot come from
a valid proof — a proof of length L means the shortest is at most L — so those
rows can only be a truncated principal variation. Topaz reports
`principal_variation().len()` under df-pn, which is neither the shortest proof
nor reliably complete. That single artefact explains the whole distribution,
including the rarer rows that overstate (df-pn returns *a* proof, not the
shortest) and the 1,013 rows with an impossible even length.

## What it does

Solves every distinct position across `puzzles`, `failed_puzzles` and
`topaz_missed_tinues` (113,667 of them) under both scopes, and records the
results in a new `syntaks_solves` table. `puzzles.tinue_length` is then
rewritten from the tak-chain result — the scope Topaz used, so the column
keeps its original meaning — with the previous value preserved in a new
`tinue_length_topaz` column. Nothing is destroyed.

Full-scope results live in their own columns rather than driving
`tinue_length`, because full scope also finds *gap tinues* (wins whose first
move threatens nothing), which Topaz cannot see by construction. Where full
finds a win and tak-chain does not, `gap_tinue` is set.

## Safety

The database lives on a fuseblk mount (`~/MEGA`), where SQLite's locking is
not trustworthy, so all work happens on a local copy and is only written back
after an integrity check. Run with `--install` to perform that final copy.

Resumable: every chunk is committed as it completes, and work is selected by
"no row yet for this scope", so re-running continues where it stopped.

    tools/verify-puzzle-db.py --pass chain        # ~1 hour
    tools/verify-puzzle-db.py --pass full         # several hours
    tools/verify-puzzle-db.py --apply             # rewrite tinue_length
    tools/verify-puzzle-db.py --install           # copy back to MEGA
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = pathlib.Path(__file__).resolve().parent.parent
TINUE = REPO / "target" / "release" / "tinue"
LIVE = pathlib.Path.home() / "MEGA" / "PTN" / "puzzles" / "puzzles.db"
WORK = pathlib.Path.home() / "Projects" / "tak-puzzles" / "work" / "puzzles.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS syntaks_solves (
    tps           TEXT PRIMARY KEY,
    size          INTEGER,
    sources       TEXT,
    chain_verdict TEXT, chain_plies INTEGER, chain_pv TEXT,
    chain_nodes   INTEGER, chain_max_plies INTEGER, chain_ms REAL,
    full_verdict  TEXT, full_plies INTEGER, full_pv TEXT,
    full_nodes    INTEGER, full_max_plies INTEGER, full_ms REAL,
    gap_tinue     INTEGER,
    road_distance INTEGER,
    engine        TEXT,
    verified_at   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_solves_chain ON syntaks_solves(chain_verdict, chain_plies);
CREATE INDEX IF NOT EXISTS idx_solves_gap   ON syntaks_solves(gap_tinue);
"""


def ensure_work_copy(force: bool = False) -> None:
    WORK.parent.mkdir(parents=True, exist_ok=True)
    if force or not WORK.exists():
        print(f"copying {LIVE} -> {WORK}", file=sys.stderr)
        shutil.copy2(LIVE, WORK)
    con = sqlite3.connect(WORK)
    con.executescript(SCHEMA)
    # Preserve the original labels before anything can overwrite them.
    cols = {r[1] for r in con.execute("pragma table_info(puzzles)")}
    if "tinue_length_topaz" not in cols:
        print("adding puzzles.tinue_length_topaz (copy of original labels)", file=sys.stderr)
        con.execute("ALTER TABLE puzzles ADD COLUMN tinue_length_topaz INTEGER")
        con.execute("UPDATE puzzles SET tinue_length_topaz = tinue_length")
    con.commit()
    con.close()


def seed_positions() -> None:
    """Insert one row per distinct position, tagged with where it came from."""
    con = sqlite3.connect(WORK)
    n0 = con.execute("select count(*) from syntaks_solves").fetchone()[0]
    con.execute(
        """
        INSERT OR IGNORE INTO syntaks_solves (tps, size, sources)
        SELECT u.tps,
               (length(u.tps) - length(replace(u.tps, '/', ''))) + 1,
               u.src
        FROM (
            SELECT tps, 'puzzles' src FROM puzzles
            UNION ALL SELECT tps, 'failed' FROM failed_puzzles
            UNION ALL SELECT tps, 'topaz_missed' FROM topaz_missed_tinues
        ) u
        """
    )
    con.commit()
    n1 = con.execute("select count(*) from syntaks_solves").fetchone()[0]
    print(f"positions seeded: {n1} ({n1 - n0} new)", file=sys.stderr)
    con.close()


def pending(scope: str, limit: int | None) -> list[tuple[str, int, int | None]]:
    """Positions with no result yet for this scope, with their label as a depth hint."""
    col = "chain_verdict" if scope == "tak-chain" else "full_verdict"
    con = sqlite3.connect(f"file:{WORK}?mode=ro", uri=True)
    if scope == "tak-chain":
        # Labelled rows first: those are the ones with a claim to correct, so
        # they yield results sooner and a partial run is still useful. The
        # unlabelled bulk (mostly tiltak "best move" puzzles that are not
        # tinues at all) is swept afterwards.
        order = ("(p.tinue_length_topaz IS NULL), s.size, "
                 "COALESCE(p.tinue_length_topaz, 0), s.tps")
    else:
        # Full scope only *adds* information where tak-chain came up empty:
        # restricted results are a subset of full, so a position tak-chain
        # already proved is a known tinue either way. The rows where chain
        # said no_tinue are exactly where gap tinues hide, so they go first —
        # which means an interrupted full pass has still covered the half
        # that matters.
        order = ("(s.chain_verdict IS NOT 'no_tinue'), s.size, "
                 "COALESCE(p.tinue_length_topaz, 0), s.tps")
    q = f"""
        SELECT s.tps, s.size, p.tinue_length_topaz
        FROM syntaks_solves s
        LEFT JOIN puzzles p ON p.tps = s.tps
        WHERE s.{col} IS NULL
        ORDER BY {order}
    """
    if limit:
        q += f" LIMIT {limit}"
    rows = con.execute(q).fetchall()
    con.close()
    return rows


def depth_for(scope: str, label: int | None, base: int, cap: int,
              unlabelled: int) -> int:
    """Depth budget for one position.

    Labelled rows are searched past their label — it understates, so stopping
    at it would just reproduce the bug being corrected.

    Unlabelled rows get their own, shallower budget. They are rows the
    original pipeline already ran Topaz over and found nothing, so a deep
    tak-chain re-run mostly reconfirms Topaz at great expense (962 ms per
    position at depth 13 versus 135 ms at depth 9). What actually adds
    knowledge there is the full-scope pass, which sees gap tinues Topaz
    cannot — so the budget is better spent on that.
    """
    if scope != "tak-chain":
        return base
    if not label:
        return unlabelled
    return min(cap, max(base, label + 4))


def run_chunk(chunk: list[tuple[str, int, int | None]], scope: str,
              depth: int, max_nodes: int, tt_bits: int) -> list[tuple]:
    stdin = "".join(f"{sz} {tps}\n" for tps, sz, _ in chunk)
    try:
        out = subprocess.run(
            [str(TINUE), "--batch", "--scope", scope, "--max-plies", str(depth),
             "--max-nodes", str(max_nodes), "--tt-bits", str(tt_bits)],
            input=stdin, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  chunk failed: {e.stderr[:300]}", file=sys.stderr)
        return []
    res = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    if len(res) != len(chunk):
        print(f"  chunk length mismatch {len(res)} != {len(chunk)}; skipping",
              file=sys.stderr)
        return []
    rows = []
    now = int(time.time())
    for (tps, _sz, _lab), r in zip(chunk, res):
        if "verdict" not in r:
            continue
        plies = r["plies"] if r["verdict"] == "tinue" else None
        rows.append((r["verdict"], plies, json.dumps(r.get("pv", [])),
                     r.get("nodes"), depth, r.get("ms"), r.get("road_distance"),
                     now, tps))
    return rows


def write_rows(rows: list[tuple], scope: str) -> None:
    if not rows:
        return
    pre = "chain" if scope == "tak-chain" else "full"
    con = sqlite3.connect(WORK, timeout=60)
    con.execute("pragma journal_mode=WAL")
    con.executemany(
        f"""UPDATE syntaks_solves SET {pre}_verdict=?, {pre}_plies=?, {pre}_pv=?,
            {pre}_nodes=?, {pre}_max_plies=?, {pre}_ms=?, road_distance=?,
            verified_at=?, engine='syntaks' WHERE tps=?""", rows)
    con.commit()
    con.close()


def do_pass(scope: str, base: int, cap: int, unlabelled: int, max_nodes: int,
            tt_bits: int, workers: int, chunk_size: int, limit: int | None) -> None:
    work = pending(scope, limit)
    if not work:
        print(f"{scope}: nothing pending", file=sys.stderr)
        return
    print(f"{scope}: {len(work)} positions pending", file=sys.stderr)

    # Group by (size, depth) so every chunk is homogeneous: the solver keeps
    # board size in a process-global atomic, and one depth per invocation.
    buckets: dict[tuple[int, int], list] = {}
    for tps, size, label in work:
        d = depth_for(scope, label, base, cap, unlabelled)
        buckets.setdefault((size, d), []).append((tps, size, label))

    # Chunk size shrinks with depth. A deep bucket can take ~13 s per position,
    # so a flat 400 would mean a single chunk running for well over an hour —
    # far too coarse to checkpoint against, and far too coarse to report on.
    def sized(depth: int) -> int:
        return max(20, chunk_size // max(1, 2 ** ((depth - 9) // 4)))

    chunks = [(list(v[i:i + sized(k[1])]), k[1])
              for k, v in buckets.items()
              for i in range(0, len(v), sized(k[1]))]
    print(f"{scope}: {len(chunks)} chunks over {len(buckets)} (size,depth) buckets",
          file=sys.stderr)

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_chunk, c, scope, d, max_nodes, tt_bits): len(c)
                for c, d in chunks}
        # as_completed, not submission order: awaiting in order would leave
        # finished chunks unwritten behind one slow chunk, so a crash could
        # discard hours of completed work and progress would read as stalled.
        for fut in as_completed(futs):
            write_rows(fut.result(), scope)
            done += futs[fut]
            el = time.time() - t0
            rate = done / el if el else 0
            eta = (len(work) - done) / rate / 60 if rate else 0
            print(f"  {done}/{len(work)}  {rate:.1f}/s  eta {eta:.0f}m", flush=True)


def requeue_aborted(scope: str) -> None:
    """Clear results for rows that hit the node cap so they are searched again.

    Aborted rows are the whole reason a two-tier strategy works: a cheap cap
    resolves ~99% of positions quickly and parks the rest honestly, then this
    requeues just that tail for a run with a real budget. Because an aborted
    row never overwrites `tinue_length`, nothing incorrect is published in the
    meantime.
    """
    pre = "chain" if scope == "tak-chain" else "full"
    con = sqlite3.connect(WORK)
    n = con.execute(
        f"select count(*) from syntaks_solves where {pre}_verdict = 'aborted'").fetchone()[0]
    con.execute(
        f"""UPDATE syntaks_solves SET {pre}_verdict=NULL, {pre}_plies=NULL, {pre}_pv=NULL,
            {pre}_nodes=NULL, {pre}_max_plies=NULL, {pre}_ms=NULL
            WHERE {pre}_verdict = 'aborted'""")
    con.commit()
    con.close()
    print(f"requeued {n} aborted {scope} rows", file=sys.stderr)


def apply_labels() -> None:
    """Rewrite puzzles.tinue_length from the verified tak-chain result."""
    con = sqlite3.connect(WORK)
    con.execute("""
        UPDATE syntaks_solves SET gap_tinue =
            CASE WHEN full_verdict='tinue' AND chain_verdict='no_tinue' THEN 1
                 WHEN full_verdict IS NOT NULL AND chain_verdict IS NOT NULL THEN 0
                 ELSE NULL END
    """)
    # Only touch rows we actually proved something about. An `aborted` result
    # means the budget ran out, which is not evidence of anything.
    before = con.execute(
        "select count(*) from puzzles where tinue_length is not null").fetchone()[0]
    con.execute("""
        UPDATE puzzles SET tinue_length = (
            SELECT s.chain_plies FROM syntaks_solves s WHERE s.tps = puzzles.tps
        )
        WHERE EXISTS (
            SELECT 1 FROM syntaks_solves s
            WHERE s.tps = puzzles.tps AND s.chain_verdict IN ('tinue','no_tinue')
        )
    """)
    con.commit()
    after = con.execute(
        "select count(*) from puzzles where tinue_length is not null").fetchone()[0]
    changed = con.execute("""
        select count(*) from puzzles
        where tinue_length is not tinue_length_topaz""").fetchone()[0]
    print(f"tinue_length: {before} -> {after} non-null; {changed} rows changed",
          file=sys.stderr)
    con.close()


def report() -> None:
    con = sqlite3.connect(f"file:{WORK}?mode=ro", uri=True)
    def q(sql):
        return con.execute(sql).fetchall()
    print("\n=== syntaks_solves ===")
    for label, sql in [
        ("chain verdicts", "select chain_verdict, count(*) from syntaks_solves group by 1"),
        ("full verdicts", "select full_verdict, count(*) from syntaks_solves group by 1"),
        ("gap tinues", "select gap_tinue, count(*) from syntaks_solves group by 1"),
    ]:
        print(f"{label}: {dict((str(a), b) for a, b in q(sql))}")
    print("\n=== label corrections (vs tinue_length_topaz) ===")
    rows = q("""
        select case
                 when tinue_length_topaz is null and tinue_length is not null then 'newly labelled'
                 when tinue_length is null and tinue_length_topaz is not null then 'label removed'
                 when tinue_length > tinue_length_topaz then 'was too short'
                 when tinue_length < tinue_length_topaz then 'was too long'
                 else 'unchanged' end, count(*)
        from puzzles group by 1 order by 2 desc""")
    for k, v in rows:
        print(f"  {k:>16}: {v}")
    con.close()


def install() -> None:
    con = sqlite3.connect(f"file:{WORK}?mode=ro", uri=True)
    ok = con.execute("pragma integrity_check").fetchone()[0]
    counts = {t: con.execute(f"select count(*) from {t}").fetchone()[0]
              for t in ("games", "puzzles", "failed_puzzles",
                        "topaz_missed_tinues", "tinue_followups", "road_win_followups")}
    con.close()
    if ok != "ok":
        print(f"REFUSING to install: integrity_check said {ok!r}", file=sys.stderr)
        sys.exit(1)
    live = sqlite3.connect(f"file:{LIVE}?mode=ro", uri=True)
    for t, n in counts.items():
        m = live.execute(f"select count(*) from {t}").fetchone()[0]
        if n != m:
            print(f"REFUSING to install: {t} has {n} rows, live has {m}", file=sys.stderr)
            live.close()
            sys.exit(1)
    live.close()
    print(f"integrity ok, row counts match; copying {WORK} -> {LIVE}", file=sys.stderr)
    shutil.copy2(WORK, LIVE)
    print("installed", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="which", choices=["chain", "full"])
    ap.add_argument("--apply", action="store_true", help="rewrite puzzles.tinue_length")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--install", action="store_true", help="copy work DB back over the live one")
    ap.add_argument("--refresh-copy", action="store_true", help="re-copy live DB to work (discards progress)")
    ap.add_argument("--retry-aborted", action="store_true",
                    help="requeue rows that hit the node cap, for a second pass "
                         "with a larger --max-nodes")
    ap.add_argument("--base-depth", type=int, default=13,
                    help="floor for labelled rows; the depth used by the full pass")
    ap.add_argument("--unlabelled-depth", type=int, default=9,
                    help="tak-chain depth for rows with no original label (default 9)")
    ap.add_argument("--cap-depth", type=int, default=21)
    ap.add_argument("--max-nodes", type=int, default=2_000_000)
    ap.add_argument("--tt-bits", type=int, default=22)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if not TINUE.exists():
        print(f"missing {TINUE}; run: cargo build --release --bin tinue", file=sys.stderr)
        return 2

    ensure_work_copy(args.refresh_copy)
    seed_positions()

    if args.retry_aborted:
        requeue_aborted("tak-chain" if args.which == "chain" else "full")

    if args.which:
        scope = "tak-chain" if args.which == "chain" else "full"
        do_pass(scope, args.base_depth, args.cap_depth, args.unlabelled_depth,
                args.max_nodes, args.tt_bits, args.workers, args.chunk_size,
                args.limit)
    if args.apply:
        apply_labels()
    if args.report:
        report()
    if args.install:
        install()
    return 0


if __name__ == "__main__":
    sys.exit(main())
