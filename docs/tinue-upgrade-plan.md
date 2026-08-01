# Syntaks tinue solver — upgrade plan

> **Status.** Upgrade 1 (tak-chain scope) is implemented on the `tinue-solver`
> branch, together with the sweep pre-filter, the native CLI, and the wasm
> scope parameter. Upgrade 2 (df-pn) and the PTN-Ninja wiring are **not** done.
> The branch rename described at the bottom is already complete.
>
> Corrections made while executing, each marked inline below: the defender
> restriction is implemented by rechecking the resulting position rather than
> by move geometry; the TT needs a scope namespace the plan didn't call for;
> and one of the two "strict, always-safe fast-outs" is not actually safe.
>
> Validation: **2,517 positions** (curated corpus, random 5×5/6×6, and
> road-distance-filtered contested positions) diffed against the
> `topaz-tinue-web` oracle — **0 verdict mismatches**. A further 1,000
> contested positions were solved under both scopes: 103 tinues found
> identically by each with matching ply counts, **0 subset violations**, and
> 0 gap tinues at depth ≤ 7 (both known gap tinues are mate-in-9, so this is
> the expected result — they are exotic). Harness:
> `tools/tinue-differential.py`, positions from `genpos`.
>
> Measured effect of the restriction on those 1,000 positions at depth 7:
> **6.3 s restricted vs 444.8 s full — a ~70× speedup.**

Two upgrades, driven by the syntaks-vs-Topaz comparison (see the `tinue-benchmark`
branch and the `../topaz-tinue-web` oracle crate). Goal: one solver that is both
**fast on strict tinues** (Topaz's strength) and **complete** (finds gap/non-tak
tinues — proven real via `morten_5s`, a mate-in-9 whose first move `c2` is quiet).

## Background: the two axes are orthogonal

Topaz's speed came from *two separable* choices; syntaks can adopt them independently.

| Axis | Effect | syntaks today | this plan |
|---|---|---|---|
| **Move set**: full ↔ tak-chain-restricted | *What* is found (semantic) | full only | add restricted mode |
| **Algorithm**: iterative-deepening ↔ df-pn | *How fast* (perf) | ID only | add df-pn |

- **Restricting the move set** (attacker → only tak threats; defender → only threat responses) finds *only* strict tak-chain tinues but collapses branching. This is a **semantic** change — user-visible in *what counts as tinue*.
- **df-pn** finds the same set as ID but explores best-first; big win on the wide/deep full-move search. This is a **performance** change — with one caveat: df-pn proves *existence*, not the *shortest* mate.

`{ID, df-pn} × {full, restricted}` = four valid configurations. The two use cases pick different corners:

- **Full-game sweep** (mark tak/tinue per ply): **restricted + ID**. Restriction = the conventional tinue mark and keeps branching tiny; ID gives the exact short mate distance the UI shows. Fast enough without df-pn.
- **Deep single-position** ("is there a forced win here?"): **full + df-pn**. Completeness (gap tinues) with df-pn taming the branching; recover shortest distance with a bounded ID pass only if the UI needs it.

---

## Upgrade 1 — tak-chain-restricted move generation (semantic mode)

Add a restricted mode to the solver, selected via `Limits`.

### Solver changes (`src/tinue.rs`)
- Add `pub enum TinueScope { Full, TakChain }` (or `restrict_tak_chain: bool`) to `Limits`.
- **TT namespacing (not in the original plan, required for soundness).** A `TakChain` `NoWin` entry is the *weaker* claim "no tak-chain win at this depth"; a `Full` `NoWin` means "no win at all". Because the sweep shares one `Tt` across many `solve_with_tt` calls, a full search could probe a restricted `NoWin` and discard exactly the gap tinues it exists to find. Scope is therefore XOR'd into the TT key alongside the existing attacker mask, keeping the two namespaces disjoint. `score_moves` takes a `scope` for the same reason.
- **Attacker nodes** (`search_attacker`): when restricted, filter generated moves to those that create a tak (road-in-1) threat. Need a `tak_threats(pos)` helper: for each attacker move, does the resulting position give the attacker a `has_road`-in-one continuation? (Topaz's `get_tak_threats` is the reference; syntaks has `has_road` to build on.) If no tak threats → this node is a loss (`DefenderHolds`), mirroring Topaz's `NoTakThreats`.
- **Defender nodes** (`search_defender`): when restricted, discard replies that fail to address the attacker's road threat. **Implemented as verify-by-recheck, not as a geometric filter**: generate every legal reply, then ask the *resulting position* whether the attacker still has a road-in-1. Deciding this from the move's shape (source/target square vs. a "road-relevance zone") is the pruning that was reverted as unsound — it ignored a spread's intermediate drop squares, so a spread that blocked via a middle square looked irrelevant and got pruned, producing false tinues. Consulting the position instead cannot be fooled that way.
  - A discarded reply is **recorded as a 2-ply loss**, not dropped: an AND node must still account for every child, and a node where *every* reply loses this way would otherwise look childless and be misread as `DefenderHolds`.
  - Gated on `depth >= 2` so mate distances continue to match what iterative deepening reports.

### Correctness
- Restricted mode must **agree exactly with Topaz** on strict tinues. Use `../topaz-tinue-web` as the oracle: run both over the corpus + a larger random-position set; every verdict must match. This is the primary validation for Upgrade 1.
- Full mode is a *superset*: any restricted-mode tinue must also be a full-mode tinue (regression check).

### Soundness note
Restricted mode proves *"a tak-chain tinue exists"*. A restricted `no_tinue` does **not** imply no tinue (gap tinues excluded). The UI must label it accordingly (see PTN-Ninja section).

---

## Upgrade 2 — df-pn search (performance)

Add a df-pn driver alongside the existing ID driver; both consume the same move
generation (full or restricted).

### Solver changes
- New `search_dfpn` implementing phi/delta proof/disproof numbers with threshold
  re-descent, storing `(phi, delta)` in the TT instead of (or alongside) the
  current Win/NoWin-at-depth flags. This likely means a **second TT entry format**
  or a widened entry; keep the existing format for the ID path.
- The AND/OR structure is unchanged (`search_attacker`/`search_defender` roles);
  df-pn changes only *node selection and bookkeeping*, not the game logic or the
  move-set restriction — so it composes with Upgrade 1.

### The shortest-mate caveat (important)
- ID currently returns the **minimal** ply count and enumerates **all** winning
  first moves; PTN-Ninja's "Tinue in N" and full-winner marking depend on both.
- df-pn returns *a* proof, not the shortest. Policy: use df-pn to prove
  **existence** fast, then a **bounded ID pass** (depth-capped at the now-known
  proof depth) to recover the exact shortest line + all winners *only when the UI
  needs them*. Deep "is it tinue?" checks can skip this.

### Validation
- df-pn and ID must return the same verdicts on the whole corpus (existence).
- Where both complete, the recovered shortest distance must match ID's.

---

## Public surface — keep native + wasm in lockstep

The solver core stays platform-agnostic (it already compiles native and wasm).
**Native must remain first-class**, not just a benchmark afterthought.

- **Native CLI** (`src/bin/tinue.rs`, currently on `tinue-benchmark`): bring a
  cleaned version onto the main branch. Expose *all* knobs: scope (full/tak-chain),
  algorithm (id/dfpn/auto), `max_plies`, `tt_bits`, `root_move`, per-depth progress.
  This is the test/benchmark harness and must exercise every mode.
- **wasm API** (`src/wasm.rs`): extend `solve_tinue` / `TinueSolver` with `scope`
  and `algorithm` params (default `auto`). Keep the existing result shape
  (`outcome`, `nodes`); add nothing the native path can't also produce.
- Every mode reachable from both entry points so native runs stay a faithful
  proxy for wasm behavior.

---

## PTN-Ninja UX

Two user-facing knobs, both *answer-shaped*. The algorithm (ID vs df-pn) is
**not** one of them — it stays fully automatic beneath these.

1. **Scope (tak-chain vs full)** — **persistent user-facing toggle** in the
   syntaks engine settings: e.g. "Find gap tinues (slower)" on/off, defaulting per
   mode (sweep → tak-chain; deep → full). Changes *what counts as a tinue*, so the
   user must own it. When restricted mode returns no tinue, label it
   "no tak-chain tinue" — not "no tinue" — so a gap tinue isn't implied absent.
2. **Depth-of-answer — "Quick check" vs "Full solve"** — surfaced as an
   **escalation**, not a persistent setting. This is the user-meaningful
   projection of the algorithm choice (Quick check ⇒ df-pn existence proof;
   Full solve ⇒ df-pn-then-recover / ID), so the algorithm itself never appears.
   - **Quick check**: fast yes/no. Reports "Tinue — confirmed" with **no ply
     count and no PV** — df-pn's line isn't guaranteed shortest, so showing
     "Tinue in N" would be a wrong-looking answer.
   - **Full solve**: exact "Tinue in N", the shortest PV, and all winning first
     moves.
   - **Progressive disclosure**: default toward Full solve; when it's slow (or on
     a known-hard position) offer *"just confirm it's a tinue"* for the fast
     answer. The right choice depends on the position's difficulty, which the user
     only learns by trying — so escalation beats an upfront toggle. A user can get
     "yes, it's winning" in seconds even on a mate-in-13 whose full line is out of
     reach.
   - Depth-of-answer basically only matters in the **full/deep** case: in a sweep
     the mate is shallow, the minimal solve is already cheap, and you always want
     "in N" + the move mark.

Rationale: **scope** changes *what counts as tinue* (the answer's meaning) →
persistent toggle. **Depth-of-answer** changes *how much you learn about the win*
(detail + time) → escalation affordance. Both are answer-shaped; the **algorithm**
is cost-only and stays automatic beneath them. Surface answer-shaped choices; hide
cost-only ones.

Routing: the existing `tinue-annotator.js` already has `sweepGame` and
`searchPosition` — wire scope + quick/full defaults per entry point there
(sweep → tak-chain + full-but-shallow; deep → full + Quick-check-then-escalate),
with the scope toggle overriding the sweep default when the user opts in.

---

## Sweep pre-filter — skip positions that can't be tinue

During a full-game sweep most positions (especially the opening) are nowhere near
a road and shouldn't cost a solver call at all. The skip test must be
**position-intrinsic** — NOT move number/index, since a game may start from an
arbitrary initial TPS (so ply count carries no information about road proximity).

**Signal: attacker road-distance.** A cheap flood-fill over the board: how many
more squares must the attacker come to control to complete the nearest road?
(attacker road-squares = cost 0, empty = 1, opponent-blocked = impassable; take the
min-cost crossing over both axes.) Reuse the existing road/bitboard machinery
(`has_road`, the SIMD road kernel) rather than a fresh graph.

**Rule (sweep only):** if `road_distance > attacker_moves_in_horizon + margin`,
skip and report no_tinue without invoking the search.

**Soundness.** road-distance counts *placements*, but one spread can fill several
path squares in a single move — so it is a **heuristic lower bound, not a strict
one**. Therefore:
- apply it ONLY to the sweep (best-effort marking), **never** to an explicit
  Quick-check / Full-solve on a single position (there, correctness > speed);
- keep a conservative `margin` to absorb spread acceleration (tunable; default
  generous). A tinue it could miss would be an exotic deep *opening* forced win
  that the shallow sweep horizon wouldn't reach anyway.

**Strict, always-safe fast-outs.** Only one of the two originally listed here
survives scrutiny:

- *The position already has a road* — safe, and already implemented (a finished
  game is not a tinue candidate).
- ~~*The attacker has no unblocked connecting path across either axis*~~ — **not
  safe.** Impassability is not permanent: the defender may move a blocking wall
  or capstone away of their own accord, after which the square becomes
  reachable. "No path along currently-open lines" therefore does not imply "no
  road is possible in any continuation", so this cannot gate a Full-solve. It is
  folded into the heuristic pre-filter instead, where `road_distance` returning
  `None` counts as "far away" like any other over-horizon verdict.

**Composition.** Complements the sweep's existing backward-iteration
proven/no-tinue cache: the cache short-circuits repeated positions; the pre-filter
avoids the solver entirely on clearly-hopeless ones. (Don't assume road-distance
is monotonic across the game — captures/spreads can push it back up — so compute
it per position.)

---

## Testing strategy

- `../topaz-tinue-web` = correctness oracle for restricted mode (must match exactly).
- Cross-checks: restricted ⊆ full; df-pn verdicts == ID verdicts; recovered
  shortest == ID shortest.
- Keep `morten_5s` (mate-in-9 gap tinue) as the canonical full-mode-only case:
  full mode finds it, restricted mode must report "no tak-chain tinue".
- Random-position differential testing native (fast, no wasm round-trip).

---

## Sequencing

1. **Upgrade 1 (restricted mode)** first — highest value (sweep speed + the
   user-facing capability), lower risk, validated against Topaz. Ship the scope
   toggle end-to-end (solver → wasm → PTN-Ninja) before touching the algorithm.
2. **Upgrade 2 (df-pn)** second — larger, independent; needed only to make the
   *full* deep search fast. Land behind `algorithm=auto` with ID as the safe
   default and df-pn opt-in until the shortest-recovery pass is proven.

The **sweep pre-filter** rides with Upgrade 1 (it's a sweep optimization, and the
sweep is where restricted scope also defaults on). The **Quick-check vs Full-solve**
escalation depends on df-pn's existence proof, so it lands with Upgrade 2.

## Branch rename

`wasm_tinue` now covers native + wasm; the name is misleading. Rename to
`tinue-solver` (or `tinue`). It tracks `gruppler/wasm_tinue`, so: `git branch -m`,
push the new name, update the PR/upstream, delete the old remote branch. Do this
as a discrete step (not mid-feature) to avoid disrupting open work.

## Open questions

- df-pn TT: widen the existing entry vs. a parallel table? (affects memory + the
  shared-TT sweep path.)
- Does restricted mode need its own `winning_first_moves` semantics, or is the
  full-mode enumeration reused?
- GHI (graph-history-interaction) handling in df-pn — relevant if positions
  transpose across different histories; scope the risk before implementing.

---

## Appendix: PNS / df-pn primer (clean-room reference)

Implement df-pn from the **published algorithm**, not from Topaz's `proof.rs`
(tak-chain restriction is domain knowledge; df-pn is public). Do the coding in a
session that has never read `proof.rs`.

References: L. V. Allis, *Searching for Solutions in Games and Artificial
Intelligence* (1994) — PNS. A. Nagai, *Df-pn Algorithm for Searching AND/OR Trees
and Its Applications* (2002) — df-pn.

### AND/OR tree
- **OR** node (attacker to move): proven if *any* child proven (need one winner).
- **AND** node (defender to move): proven if *all* children proven (refute every reply).

### Proof / disproof numbers
Each node carries `(pn, dn)` — `pn` = min unexpanded leaves to **prove** it, `dn` =
min to **disprove** it.
- Leaf init: unexpanded `(1,1)`; proven `(0,∞)`; disproven `(∞,0)`.
- **OR:** `pn = min(children pn)`, `dn = Σ(children dn)`
- **AND:** `pn = Σ(children pn)`, `dn = min(children dn)`

### PNS loop
Descend from the root to the *most-proving node* (min `pn` at OR, min `dn` at AND)
→ expand that leaf → propagate `(pn,dn)` back up. Stop when root `pn = 0` (proven)
or `dn = 0` (disproven). Effect: effort flows into the narrowest, most-forcing part
of the tree — the AND-node `dn = min` rule dives few-reply defender lines
automatically.

### df-pn (the version to build)
Depth-first reformulation: pass `pn`/`dn` **thresholds** down; stay in a subtree
while its numbers stay under threshold, else return and let the parent re-pick.
State lives in the **TT**, not an explicit tree → bounded memory; provably expands
the same nodes as PNS. Structurally ≈ the existing recursive AND/OR search with
thresholds replacing the depth limit and `(pn,dn)` replacing win/nowin-at-depth.

### vs iterative deepening
- **ID** picks by **depth** (uniform, shallowest first) → finds the **shortest**
  proof, needs a depth bound, re-searches lower depths.
- **df-pn** picks by **proof difficulty** (non-uniform) → finds *a* proof at any
  depth, no bound, no uniform-breadth re-walk. Wins on wide/deep trees; loses
  shortest-mate.
- Hence the hybrid: df-pn proves existence; a bounded ID pass recovers the shortest
  line + all winners, sharing proven results through the TT.

### Hazards to design for up front
- **GHI (graph-history interaction):** a position-keyed TT entry can be *wrong*
  when the path/history matters (repetition / threefold-like states — the same
  board can be a win via one history and not another). Guard history-sensitive
  entries; don't trust a bare position hash for repetition-dependent results.
- **Seesaw / "1+ε" thrash:** naive threshold updates ping-pong between two balanced
  subtrees. Standard fix is the small-epsilon threshold bump — expected behavior,
  not a perf bug.
- **Unified TT entry:** share a "solved: win(plies?)/loss" bit between df-pn and ID
  (the reuse currency that makes Quick-check → Full-solve escalation cheap), plus
  df-pn's `(pn,dn)` scratch. Resolves the TT open question above.
