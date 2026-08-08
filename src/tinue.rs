/*
 * syntaks, a TEI Tak engine
 * Copyright (c) 2026 Ciekce
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! Tinue solver.
//!
//! A tinue is a position where the side to move (the *attacker*) has a forced
//! road win — every defender response leads to a position where the attacker
//! still has a forced road win, terminating in an actual road. This module
//! finds tinues by alpha-beta search with a binary "is there a road?" leaf
//! check, no static evaluation, and iterative deepening over odd plies.
//!
//! This is *not* general game-tree search. It assumes only road wins matter:
//! flat-count wins, draws by piece exhaustion, and any heuristic eval are
//! ignored. The attacker proves a forced road; the defender disproves it.

use crate::bitboard::Bitboard;
use crate::board::{FlatCountOutcome, Position};
use crate::core::{Direction, PieceType, Player};
use crate::movegen::generate_moves;
use crate::takmove::Move;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const TT_DEFAULT_BITS: u32 = 20;

const TT_FLAG_NOWIN: u16 = 0x4000;
const TT_FLAG_WIN: u16 = 0x8000;
const TT_FLAG_MASK: u16 = 0xC000;
const TT_VALUE_MASK: u16 = 0x3FFF;

#[derive(Copy, Clone, Default)]
struct TtEntry {
    key: u64,
    // top 2 bits: 00=empty, 01=NoWin, 10=Win. low 14 bits: plies (Win) or
    // searched depth (NoWin).
    flags: u16,
    best_move: u16,
}

/// Transposition table for the tinue search. Reusable across calls — pass
/// the same `Tt` to multiple `solve_with_tt` invocations to share the cache
/// (e.g. when sweeping a whole game). Sized to `1 << bits` entries × 16 B.
/// Memo for "does the side to move have a road-in-1 here?", the question
/// [`Searcher::road_in_1_move`] answers.
///
/// `key == 0` marks an unused slot; `mv == 0` is a cached *negative* (no such
/// move exists), which is the common and expensive answer to recompute since
/// it means every move was tried and rejected.
#[derive(Copy, Clone, Default)]
struct TakEntry {
    key: u64,
    mv: u16,
}

pub struct Tt {
    entries: Vec<TtEntry>,
    mask: usize,
    /// See [`TakEntry`]. Lives in the `Tt` rather than the searcher because a
    /// searcher is rebuilt for every iterative-deepening iteration, and this
    /// table is far too large to reallocate per depth.
    tak: Vec<TakEntry>,
    tak_mask: usize,
}

impl Tt {
    pub fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        // Deliberately small — 4 MB, not a fraction of the main table.
        //
        // Measured on deep positions this memo is worth about 1.1x, far less
        // than expected, and the reason caps its useful size: the queries do
        // not transpose. At an attacker node the question is asked about
        // `child.apply_nullmove()` for each child, and those keys are unique
        // per child, so every query inside a single iteration is a first
        // visit. The only reuse is across iterative-deepening iterations, and
        // the deepest iteration — which dominates the work — is all misses.
        //
        // A bigger table therefore buys almost nothing while costing real
        // memory in the browser, where this ships via wasm.
        let tak_size = 1usize << bits.clamp(10, 18);
        Self {
            entries: vec![TtEntry::default(); size],
            mask: size - 1,
            tak: vec![TakEntry::default(); tak_size],
            tak_mask: tak_size - 1,
        }
    }

    pub fn clear(&mut self) {
        for e in &mut self.entries {
            *e = TtEntry::default();
        }
        for e in &mut self.tak {
            *e = TakEntry::default();
        }
    }

    fn idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
    }

    /// `Some(answer)` on a hit, `None` when the position is not cached.
    ///
    /// The answer is a pure function of the position — the Zobrist key covers
    /// the board and the side to move, and reserves are derived from the
    /// board — so no scope or attacker namespacing is needed here, unlike the
    /// main table whose verdicts are relative to the searcher.
    fn probe_tak(&self, key: u64) -> Option<Option<Move>> {
        let e = self.tak[(key as usize) & self.tak_mask];
        if e.key == key && key != 0 {
            Some(Move::from_raw(e.mv))
        } else {
            None
        }
    }

    fn store_tak(&mut self, key: u64, mv: Option<Move>) {
        let idx = (key as usize) & self.tak_mask;
        self.tak[idx] = TakEntry {
            key,
            mv: mv.map_or(0, |m| m.raw()),
        };
    }

    fn probe(&self, key: u64) -> Option<TtEntry> {
        let e = self.entries[self.idx(key)];
        if e.key == key && (e.flags & TT_FLAG_MASK) != 0 {
            Some(e)
        } else {
            None
        }
    }

    fn store_win(&mut self, key: u64, plies: u32, best_move: u16) {
        let idx = self.idx(key);
        let plies = (plies.min(TT_VALUE_MASK as u32)) as u16;
        self.entries[idx] = TtEntry {
            key,
            flags: TT_FLAG_WIN | plies,
            best_move,
        };
    }

    fn store_nowin(&mut self, key: u64, depth: u32, best_move: u16) {
        let idx = self.idx(key);
        let existing = self.entries[idx];
        // Never demote a Win to a NoWin: Win is a finished proof regardless
        // of depth, NoWin is only valid up to the depth searched.
        if existing.key == key && (existing.flags & TT_FLAG_WIN) != 0 {
            return;
        }
        let depth = (depth.min(TT_VALUE_MASK as u32)) as u16;
        self.entries[idx] = TtEntry {
            key,
            flags: TT_FLAG_NOWIN | depth,
            best_move,
        };
    }
}

/// Per-search XOR mask that segregates TT entries by which player is the
/// attacker. The position's Zobrist key already encodes whose turn it is, but
/// stored Win/NoWin flags carry meaning relative to the SEARCHER'S attacker
/// — not whose turn it is in the position. Two solves with different
/// attackers can otherwise probe each other's entries (e.g. when sweeping
/// backwards through a game with a shared TT) and read winning entries with
/// the wrong winner. XORing this mask in before lookup keeps the namespaces
/// separate. Top bit chosen because syntaks's Zobrist keys leave it
/// unconstrained.
const ATTACKER_KEY_MASK_P1: u64 = 0;
const ATTACKER_KEY_MASK_P2: u64 = 1u64 << 63;

/// Second namespace axis, for the same reason as the attacker mask: a stored
/// verdict means something different under each [`TinueScope`].
///
/// A `TakChain` `NoWin` is the *weaker* claim "no tak-chain win at this
/// depth"; a `Full` `NoWin` is "no win at all at this depth". Letting a full
/// search read a restricted `NoWin` would silently discard exactly the gap
/// tinues full mode exists to find. (`Win` entries are compatible in the
/// other direction — a restricted proof is a real proof — but the two are
/// kept fully disjoint rather than relying on flag-by-flag reasoning.)
///
/// This matters in practice because the sweep path shares one [`Tt`] across
/// many `solve_with_tt` calls, and the scope toggle can flip between them.
const SCOPE_KEY_MASK_FULL: u64 = 0;
const SCOPE_KEY_MASK_TAK_CHAIN: u64 = 1u64 << 62;

#[inline]
fn attacker_key_mask(attacker: Player) -> u64 {
    match attacker {
        Player::P1 => ATTACKER_KEY_MASK_P1,
        Player::P2 => ATTACKER_KEY_MASK_P2,
    }
}

#[inline]
fn scope_key_mask(scope: TinueScope) -> u64 {
    match scope {
        TinueScope::Full => SCOPE_KEY_MASK_FULL,
        TinueScope::TakChain => SCOPE_KEY_MASK_TAK_CHAIN,
    }
}

/// Combined TT namespace for a search. See [`attacker_key_mask`] and
/// [`scope_key_mask`].
#[inline]
fn namespace_key_mask(attacker: Player, scope: TinueScope) -> u64 {
    attacker_key_mask(attacker) ^ scope_key_mask(scope)
}

/// Result of a tinue search.
#[derive(Clone, Debug)]
pub enum TinueResult {
    /// Forced road win in `plies` ply (always odd: attacker plays the last
    /// move). `pv` is one principal variation; non-PV defender moves also
    /// lose, but only the longest defense's continuation is recorded.
    ///
    /// `winning_first_moves` lists every attacker move at the root that
    /// leads to a proven forced road win at the same depth. Always contains
    /// at least `pv[0]`. Populated when `Limits::find_all_winners` is set
    /// (default true), allowing callers — e.g. an auto-annotator — to mark
    /// every move that's "on the road to tinue" rather than only the
    /// engine's first-found principal variation.
    Tinue {
        plies: u32,
        pv: Vec<Move>,
        winning_first_moves: Vec<Move>,
    },

    /// No tinue exists within the depth limit. Either the attacker has no
    /// forced win up to `searched_plies` (proven), or the defender has a
    /// concrete refutation found during the search.
    NoTinue { searched_plies: u32 },

    /// Search aborted before completing — time, node, or external cancel.
    Aborted { reason: AbortReason, searched_plies: u32 },
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AbortReason {
    Cancelled,
    Nodes,
    Depth,
}

/// Which move set the search explores — a **semantic** choice that changes
/// *what counts as a tinue*, not merely how fast it is found.
///
/// * [`TinueScope::Full`] searches every legal move for both sides. It finds
///   every forced road win, including *gap tinues* whose winning line passes
///   through a quiet move that threatens nothing (`archvenison_2026_05_24`
///   and `morten_5s_tinue_2` in this module's tests are both mate-in-9 gap
///   tinues).
/// * [`TinueScope::TakChain`] restricts the attacker to moves that leave a
///   live road-in-1 threat, and the defender to replies that answer it. This
///   is the conventional "tinue" of Tak literature and tooling (it is what
///   Topaz proves), and it collapses the branching factor enormously — but
///   it is *incomplete*: a `NoTinue` under this scope means "no tak-chain
///   tinue", **not** "no tinue". Callers must label it accordingly.
///
/// Restricted results are a strict subset of full results: anything provable
/// under `TakChain` is provable under `Full`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum TinueScope {
    #[default]
    Full,
    TakChain,
}

impl TinueScope {
    /// Canonical spelling, used by the CLI, the TEI `tinue` command and the
    /// wasm bindings so every entry point names the modes identically.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            TinueScope::Full => "full",
            TinueScope::TakChain => "tak-chain",
        }
    }
}

impl std::str::FromStr for TinueScope {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "full" => Ok(TinueScope::Full),
            "tak-chain" | "tak_chain" | "takchain" | "chain" | "tak" => Ok(TinueScope::TakChain),
            _ => Err(()),
        }
    }
}

#[derive(Clone)]
pub struct Limits<'a> {
    pub max_plies: u32,
    pub max_nodes: u64,
    pub cancel: Option<&'a AtomicBool>,
    /// When set, after proving a tinue the solver enumerates the remaining
    /// root attacker moves to find any others that also win at the proven
    /// depth. Cheap because the TT is hot from the primary search. The
    /// extra winners (if any) land in `TinueResult::Tinue::winning_first_moves`.
    pub find_all_winners: bool,
    /// Move-set restriction. See [`TinueScope`]. Defaults to `Full` so that
    /// existing callers keep the complete (gap-tinue-finding) semantics.
    pub scope: TinueScope,
    /// **Sweep-only** heuristic pre-filter. When `Some(margin)`, a position
    /// whose attacker [`road_distance`] exceeds the attacker's move budget
    /// within `max_plies` plus `margin` is reported `NoTinue` without any
    /// search at all — most of a game (all of the opening) is nowhere near a
    /// road and should not cost a solver call.
    ///
    /// The test is position-intrinsic rather than ply-index based on purpose:
    /// a game may start from an arbitrary TPS, so move number carries no
    /// information about road proximity.
    ///
    /// [`road_distance`] is a heuristic, not a lower bound (a spread can fill
    /// several path squares in one move, and blockers are not permanent), so
    /// this can in principle skip a real tinue. Leave it `None` — the default
    /// — for any explicit single-position solve, where correctness outranks
    /// speed. It exists for best-effort full-game marking only.
    pub prefilter_margin: Option<u32>,
}

impl Default for Limits<'_> {
    fn default() -> Self {
        Self {
            max_plies: 21,
            max_nodes: u64::MAX,
            cancel: None,
            find_all_winners: true,
            scope: TinueScope::Full,
            prefilter_margin: None,
        }
    }
}

/// Would the sweep pre-filter skip this position — i.e. is the side to move
/// too far from any road for a win within `max_plies` to be plausible?
///
/// Heuristic. See [`Limits::prefilter_margin`] and [`road_distance`] for why
/// this must not gate an explicit single-position solve.
#[must_use]
pub fn prefilter_skips(pos: &Position, max_plies: u32, margin: u32) -> bool {
    // Attacker moves available inside an odd-ply horizon: plies 1, 3, 5, …
    let attacker_moves = max_plies.div_ceil(2);
    match road_distance(pos, pos.stm()) {
        None => true,
        Some(d) => d > attacker_moves + margin,
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub nodes: u64,
    pub max_depth_reached: u32,
}

struct Searcher<'a, 'b> {
    attacker: Player,
    /// XOR'd into every TT key so entries are partitioned by attacker and
    /// scope. See [`namespace_key_mask`] for the rationale.
    ns_mask: u64,
    scope: TinueScope,
    nodes: AtomicU64,
    node_limit: u64,
    cancel: Option<&'a AtomicBool>,
    aborted: bool,
    tt: &'b mut Tt,
    /// Scratch move buffer reused by [`Searcher::has_road_in_1`]. That helper
    /// is called once per move at every restricted node, so allocating a
    /// fresh `Vec` per call would dominate the search. It never recurses, so
    /// a single buffer swapped out with `mem::take` is safe.
    threat_buf: Vec<Move>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum NodeOutcome {
    /// Attacker forces a road win in this many plies (odd from an
    /// attacker-to-move node, even from a defender-to-move node).
    AttackerWins(u32),
    /// Defender survives the depth limit (or refutes outright).
    DefenderHolds,
    /// Search aborted in this subtree.
    Aborted,
}

impl<'a, 'b> Searcher<'a, 'b> {
    fn new(attacker: Player, limits: &Limits<'a>, tt: &'b mut Tt) -> Self {
        Self {
            attacker,
            ns_mask: namespace_key_mask(attacker, limits.scope),
            scope: limits.scope,
            nodes: AtomicU64::new(0),
            node_limit: limits.max_nodes,
            cancel: limits.cancel,
            aborted: false,
            tt,
            threat_buf: Vec::with_capacity(128),
        }
    }

    #[inline]
    fn tt_key(&self, pos: &Position) -> u64 {
        pos.key() ^ self.ns_mask
    }

    #[inline]
    fn restricted(&self) -> bool {
        self.scope == TinueScope::TakChain
    }

    /// The side to move at `pos` completes their own road with this move —
    /// i.e. `pos` is "tak" for them. Returns the completing move so callers
    /// can record an exact PV; `None` if no such move exists.
    ///
    /// No move is ever applied. A flat or capstone on an empty square `s`
    /// changes the mover's road bitboard to exactly `roads | s`, so one
    /// `has_road` on that union settles a placement; walls never extend a
    /// road and are skipped outright; and spreads go through
    /// [`Position::spread_completes_road`], which derives the resulting road
    /// bitboard from the drop pattern. Every branch is now bitboard work
    /// against `pos` itself, so the whole scan touches one position.
    fn road_in_1_move(&mut self, pos: &Position) -> Option<Move> {
        // Memoised: this is called once per candidate move at every
        // restricted node, so a node costs O(moves^2) of this scan. Positions
        // repeat heavily across the search (the same reason the transposition
        // table pays off), and a cached answer is exact rather than
        // depth-bounded, so a hit is always usable.
        let key = pos.key();
        if let Some(hit) = self.tt.probe_tak(key) {
            return hit;
        }

        let stm = pos.stm();
        let roads = pos.roads(stm);

        let mut buf = std::mem::take(&mut self.threat_buf);
        generate_moves(&mut buf, pos);

        let mut found = None;
        for &mv in &buf {
            let completes = if mv.is_spread() {
                pos.spread_completes_road(mv)
            } else if mv.pt() == PieceType::Wall {
                false
            } else {
                crate::road::has_road(roads.with_sq(mv.sq()))
            };
            if completes {
                found = Some(mv);
                break;
            }
        }

        self.threat_buf = buf;
        self.tt.store_tak(key, found);
        found
    }

    /// Does the attacker threaten to complete a road on their next move, in a
    /// position where the *defender* is to move? Asked at attacker nodes of a
    /// restricted search to decide whether a candidate move keeps the tak
    /// chain alive.
    ///
    /// Implemented by passing the turn back to the attacker with a null move.
    /// That is exactly the "if the defender did nothing, could I finish?"
    /// question a tak threat encodes.
    fn attacker_threatens_road(&mut self, pos: &Position) -> bool {
        debug_assert_ne!(pos.stm(), self.attacker);
        let passed = pos.apply_nullmove();
        self.road_in_1_move(&passed).is_some()
    }

    fn check_abort(&mut self) -> bool {
        if self.aborted {
            return true;
        }
        let n = self.nodes.load(Ordering::Relaxed);
        if n >= self.node_limit {
            self.aborted = true;
            return true;
        }
        if let Some(flag) = self.cancel
            && flag.load(Ordering::Relaxed)
        {
            self.aborted = true;
            return true;
        }
        false
    }

    fn bump_nodes(&self) {
        self.nodes.fetch_add(1, Ordering::Relaxed);
    }

    /// Walk the TT from `pos`, replaying each cell's stored best_move, until
    /// `pv` reaches `target_len`, the chain breaks, or a road appears.
    fn extend_pv_via_tt(&self, pos: &Position, pv: &mut Vec<Move>, target_len: usize) {
        let mut cur = pos.clone();
        for &mv in pv.iter() {
            if !cur.is_legal(mv) {
                return;
            }
            cur = cur.apply_move(mv);
            if cur.has_road(self.attacker) || cur.has_road(self.attacker.flip()) {
                return;
            }
        }
        while pv.len() < target_len {
            let entry = match self.tt.probe(self.tt_key(&cur)) {
                Some(e) => e,
                None => return,
            };
            let mv = match Move::from_raw(entry.best_move) {
                Some(m) => m,
                None => return,
            };
            if !cur.is_legal(mv) {
                return;
            }
            pv.push(mv);
            cur = cur.apply_move(mv);
            if cur.has_road(self.attacker) || cur.has_road(self.attacker.flip()) {
                return;
            }
        }
    }

    /// Enumerate every attacker root move (other than `primary`) that also
    /// wins at the given depth. Appends to `winners` in move-generation
    /// order. Soft-bounded by the existing search limits (node budget,
    /// cancel flag) and stops early on abort. Run after a primary tinue
    /// has been proven — the TT is hot, so non-winning moves cut fast and
    /// other winners are TT hits.
    fn collect_root_winners(
        &mut self,
        pos: &Position,
        depth: u32,
        primary: Move,
        winners: &mut Vec<Move>,
    ) {
        let mut moves = Vec::with_capacity(64);
        generate_moves(&mut moves, pos);

        for &mv in &moves {
            if mv == primary {
                continue;
            }
            if self.check_abort() {
                return;
            }

            let next = pos.apply_move(mv);

            // Direct road completion.
            if next.has_road(self.attacker) {
                winners.push(mv);
                continue;
            }
            // Suicide / flat-resolution branches are not winning candidates.
            if next.has_road(self.attacker.flip()) {
                continue;
            }
            if !matches!(next.count_flats(), FlatCountOutcome::None) {
                continue;
            }
            // Same restriction as the primary search, so the enumerated
            // winners are drawn from the same move set as the proof.
            if self.restricted() && !self.attacker_threatens_road(&next) {
                continue;
            }

            let mut sub_pv = Vec::new();
            match self.search_defender(&next, depth - 1, &mut sub_pv) {
                NodeOutcome::AttackerWins(_) => winners.push(mv),
                NodeOutcome::DefenderHolds => {}
                NodeOutcome::Aborted => return,
            }
        }
    }

    /// Attacker is to move at `pos`. Search up to `depth` plies. Returns
    /// AttackerWins(k) for the *first* forced win found (≤ depth), or
    /// DefenderHolds if no forced win is provable at this depth.
    ///
    /// Note: this returns the first mate found, not the shortest. With
    /// iterative deepening over odd plies, the first time we find a mate
    /// is at the smallest odd depth at which one exists, which gives the
    /// shortest in even ply count, but the PV's plies value reflects the
    /// concrete continuation length, not the optimal one.
    fn search_attacker(&mut self, pos: &Position, depth: u32, pv: &mut Vec<Move>) -> NodeOutcome {
        debug_assert_eq!(pos.stm(), self.attacker);

        if self.check_abort() {
            return NodeOutcome::Aborted;
        }

        if depth == 0 {
            return NodeOutcome::DefenderHolds;
        }

        let key = self.tt_key(pos);
        let tt_hit = self.tt.probe(key);
        let tt_move = tt_hit.and_then(|e| Move::from_raw(e.best_move));

        if let Some(e) = tt_hit {
            let value = (e.flags & TT_VALUE_MASK) as u32;
            if (e.flags & TT_FLAG_WIN) != 0 {
                if value <= depth {
                    pv.clear();
                    if let Some(mv) = tt_move {
                        pv.push(mv);
                    }
                    return NodeOutcome::AttackerWins(value);
                }
            } else if value >= depth {
                return NodeOutcome::DefenderHolds;
            }
        }

        self.bump_nodes();

        let mut moves = Vec::with_capacity(64);
        generate_moves(&mut moves, pos);
        order_attacker_moves(pos, &mut moves, self.attacker);
        if let Some(tt_mv) = tt_move
            && let Some(idx) = moves.iter().position(|&m| m == tt_mv)
        {
            moves.swap(0, idx);
        }

        for &mv in &moves {
            let next = pos.apply_move(mv);

            // After attacker's move: if attacker has a road, that's a win.
            // Per current-player-wins rule, even if defender also has a road
            // on the resulting position, attacker (the mover) wins.
            if next.has_road(self.attacker) {
                pv.clear();
                pv.push(mv);
                self.tt.store_win(key, 1, mv.raw());
                return NodeOutcome::AttackerWins(1);
            }

            // If only defender has a road, attacker handed defender the
            // game — skip. This can happen via a wall-smash that exposes
            // a defender flat completing their road.
            if next.has_road(self.attacker.flip()) {
                continue;
            }

            // Flat-count terminal positions don't count as tinue wins for
            // either side; skip.
            if !matches!(next.count_flats(), FlatCountOutcome::None) {
                continue;
            }

            // Tak-chain scope: the attacker may only play moves that leave a
            // live road threat. A move that threatens nothing breaks the
            // chain — and those quiet moves are precisely what gap tinues
            // are built on, which is why this scope cannot find them.
            //
            // An outright road win was already returned above, so this can
            // never filter away a winning move.
            if self.restricted() && !self.attacker_threatens_road(&next) {
                continue;
            }

            let mut sub_pv = Vec::new();
            match self.search_defender(&next, depth - 1, &mut sub_pv) {
                NodeOutcome::AttackerWins(plies) => {
                    pv.clear();
                    pv.push(mv);
                    pv.extend_from_slice(&sub_pv);
                    self.tt.store_win(key, plies + 1, mv.raw());
                    return NodeOutcome::AttackerWins(plies + 1);
                }
                NodeOutcome::DefenderHolds => {}
                NodeOutcome::Aborted => return NodeOutcome::Aborted,
            }
        }

        self.tt.store_nowin(key, depth, 0);
        NodeOutcome::DefenderHolds
    }

    /// Defender is to move at `pos`. Returns AttackerWins(k) iff every legal
    /// defender move leads to an attacker forced win within depth. If any
    /// single defender move survives, returns DefenderHolds with that move
    /// in `pv` as a refutation.
    ///
    /// k is the longest forced-mate continuation found among non-refuting
    /// defender moves (defender plays the move that drags the loss out
    /// the most), but since one survival refutes outright, this is
    /// strictly informational — the actual proof obligation is "every
    /// move loses".
    fn search_defender(&mut self, pos: &Position, depth: u32, pv: &mut Vec<Move>) -> NodeOutcome {
        debug_assert_ne!(pos.stm(), self.attacker);

        if self.check_abort() {
            return NodeOutcome::Aborted;
        }

        if depth == 0 {
            return NodeOutcome::DefenderHolds;
        }

        let key = self.tt_key(pos);
        let tt_hit = self.tt.probe(key);
        let tt_move = tt_hit.and_then(|e| Move::from_raw(e.best_move));

        if let Some(e) = tt_hit {
            let value = (e.flags & TT_VALUE_MASK) as u32;
            if (e.flags & TT_FLAG_WIN) != 0 {
                if value <= depth {
                    pv.clear();
                    if let Some(mv) = tt_move {
                        pv.push(mv);
                    }
                    return NodeOutcome::AttackerWins(value);
                }
            } else if value >= depth {
                pv.clear();
                if let Some(mv) = tt_move {
                    pv.push(mv);
                }
                return NodeOutcome::DefenderHolds;
            }
        }

        self.bump_nodes();

        let mut moves = Vec::with_capacity(64);
        generate_moves(&mut moves, pos);

        if moves.is_empty() {
            return NodeOutcome::DefenderHolds;
        }

        // Geometric defender-move pruning is intentionally NOT applied here,
        // in either scope. An earlier "road-relevance zone" filter (kept the
        // move only if its source/target was in or one orthogonal step from a
        // road piece) turned out to be unsound: it ignored intermediate drop
        // squares of a spread, so a spread starting and ending outside the
        // zone but dropping a stone on a critical road-blocking square was
        // wrongly pruned. That produced false-positive Tinuës in real games.
        // Soundness > speed for tinue annotation.
        //
        // `TinueScope::TakChain` gets its branching reduction a different
        // way — see the road-in-1 shortcut inside the loop below, which
        // consults the resulting position rather than the move's shape.

        order_defender_moves(pos, &mut moves, self.attacker);
        if let Some(tt_mv) = tt_move
            && let Some(idx) = moves.iter().position(|&m| m == tt_mv)
        {
            moves.swap(0, idx);
        }

        let mut longest: Option<(u32, Vec<Move>)> = None;

        for &mv in &moves {
            let next = pos.apply_move(mv);

            // Defender played a move that handed attacker a road (e.g.,
            // smashed a wall covering attacker's flat). Per current-player-
            // wins rule, since defender is mover, defender's own road would
            // win for defender — we have to check that too.
            if next.has_road(self.attacker.flip()) {
                // defender's own road — defender wins, attacker fails to
                // achieve tinue from this branch.
                pv.clear();
                pv.push(mv);
                self.tt.store_nowin(key, depth, mv.raw());
                return NodeOutcome::DefenderHolds;
            }

            if next.has_road(self.attacker) {
                // defender's move accidentally completes attacker's road.
                // attacker wins immediately. plies-from-here = 1 (defender
                // already moved, attacker's "win" was the act of defender
                // moving). We treat this as a +0 distance: attacker doesn't
                // need to play another move. But for PV/distance counting,
                // we count it as a 1-ply win: defender's losing move.
                let cand_plies = 1;
                let take = longest.as_ref().map_or(true, |(w, _)| cand_plies > *w);
                if take {
                    longest = Some((cand_plies, vec![mv]));
                }
                continue;
            }

            // Flat-count terminal positions are not tinue endings. If
            // defender forces a flat-count draw or win, attacker has no
            // tinue here.
            match next.count_flats() {
                FlatCountOutcome::Win(p) if p == self.attacker.flip() => {
                    // defender wins flat count — attacker loses
                    pv.clear();
                    pv.push(mv);
                    self.tt.store_nowin(key, depth, mv.raw());
                    return NodeOutcome::DefenderHolds;
                }
                FlatCountOutcome::Draw => {
                    // a draw refutes tinue
                    pv.clear();
                    pv.push(mv);
                    self.tt.store_nowin(key, depth, mv.raw());
                    return NodeOutcome::DefenderHolds;
                }
                FlatCountOutcome::Win(_) => {
                    // attacker wins flat count — but tinue requires a road
                    // win, so this is also a refutation.
                    pv.clear();
                    pv.push(mv);
                    self.tt.store_nowin(key, depth, mv.raw());
                    return NodeOutcome::DefenderHolds;
                }
                FlatCountOutcome::None => {}
            }

            // Tak-chain scope: a reply that leaves the attacker a live
            // road-in-1 loses on the spot, so it needs no subtree — score it
            // as the 2-ply loss it is and move on.
            //
            // Note this *records* the move as losing rather than dropping it
            // from the move list. Dropping would be wrong twice over: the
            // AND node would silently shed a child it is still obliged to
            // refute, and a node where every reply loses this way would look
            // childless and fall through to `DefenderHolds`.
            //
            // This is the sound form of the defender pruning reverted in the
            // full search (see the note above the ordering call). It asks the
            // resulting position whether the threat actually survives instead
            // of inferring from move geometry, so a spread that blocks via an
            // intermediate drop square is classified correctly — that was the
            // exact bug that made the old zone filter produce false tinues.
            //
            // Gated on `depth >= 2` so mate distances still match what
            // iterative deepening reports: at depth 1 the child search bottoms
            // out at depth 0 and yields `DefenderHolds`, and this shortcut
            // must not claim a win the depth budget cannot pay for.
            if self.restricted() && depth >= 2
                && let Some(finish) = self.road_in_1_move(&next)
            {
                let cand_plies = 2;
                if longest.as_ref().map_or(true, |(w, _)| cand_plies > *w) {
                    longest = Some((cand_plies, vec![mv, finish]));
                }
                continue;
            }

            let mut sub_pv = Vec::new();
            match self.search_attacker(&next, depth - 1, &mut sub_pv) {
                NodeOutcome::AttackerWins(plies) => {
                    let total = plies + 1;
                    let take = longest.as_ref().map_or(true, |(w, _)| total > *w);
                    if take {
                        let mut new_pv = Vec::with_capacity(sub_pv.len() + 1);
                        new_pv.push(mv);
                        new_pv.extend_from_slice(&sub_pv);
                        longest = Some((total, new_pv));
                    }
                }
                NodeOutcome::DefenderHolds => {
                    // any single survival refutes tinue
                    pv.clear();
                    pv.push(mv);
                    self.tt.store_nowin(key, depth, mv.raw());
                    return NodeOutcome::DefenderHolds;
                }
                NodeOutcome::Aborted => return NodeOutcome::Aborted,
            }
        }

        match longest {
            Some((plies, found_pv)) => {
                let bm = found_pv.first().map(|m| m.raw()).unwrap_or(0);
                self.tt.store_win(key, plies, bm);
                *pv = found_pv;
                NodeOutcome::AttackerWins(plies)
            }
            None => {
                self.tt.store_nowin(key, depth, 0);
                NodeOutcome::DefenderHolds
            }
        }
    }
}

/// Per-move status against a warm TT, used by the UI to colour every legal
/// move in the displayed position with its tinue-relative verdict. See
/// [`score_moves`].
///
/// `plies` counts from the *current* position (before the move is played),
/// so `Win { plies: 1 }` means this move itself completes the road, and
/// `Win { plies: 3 }` means a 3-ply forced sequence starting with this move.
#[derive(Copy, Clone, Debug)]
pub enum MoveScoreKind {
    /// Attacker has a forced road win in `plies` ply from before this move.
    Win { plies: u32 },
    /// Attacker is forced to lose in `plies` ply. Only emitted when the
    /// loss is provable from the move's immediate result (board-state road
    /// or flat-count win for the defender) — TT NoWin entries are reported
    /// as `NoWin` since they only prove absence of an attacker win at a
    /// given depth, not a defender win.
    Loss { plies: u32 },
    /// Attacker has no forced win within `searched` plies from before this
    /// move. The defender may still ultimately lose at greater depth.
    NoWin { searched: u32 },
    /// Game ends after this move by flat-count resolution.
    Flat { outcome: FlatOutcome },
    /// TT has no entry for the resulting position; status is unknown
    /// without a fresh search.
    Unknown,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum FlatOutcome {
    AttackerWin,
    DefenderWin,
    Draw,
}

#[derive(Clone, Debug)]
pub struct MoveScore {
    pub mv: Move,
    pub kind: MoveScoreKind,
}

/// Score every legal move at `pos` against a warm TT, from `attacker`'s
/// perspective. Pure TT lookup — no recursive search — so this is O(legal
/// moves) and safe to call on every UI navigation tick. Moves whose
/// resulting position isn't in the TT come back as `Unknown`; the caller
/// can extend coverage with `solve_at_depth` and re-score.
///
/// `attacker` is explicit so the caller controls perspective independent
/// of whose turn it is in `pos`. Typical usage during proof exploration:
/// keep `attacker` fixed to whoever was proven to win the root puzzle and
/// pass it on every `score_moves` call as the user navigates.
///
/// `scope` must match the scope of the solve that populated `tt` — entries
/// are namespaced per scope (see [`scope_key_mask`]), so passing the wrong
/// one reports every move as `Unknown` rather than reading another scope's
/// verdicts.
pub fn score_moves(
    pos: &Position,
    attacker: Player,
    scope: TinueScope,
    tt: &Tt,
) -> Vec<MoveScore> {
    let stm = pos.stm();
    let mask = namespace_key_mask(attacker, scope);
    let mut out = Vec::with_capacity(64);
    let mut moves = Vec::with_capacity(64);
    generate_moves(&mut moves, pos);

    for mv in moves {
        let next = pos.apply_move(mv);

        // "Current player wins" rule: if the mover ended their turn with a
        // road, they win — even if the opponent also has a road on the
        // resulting board (e.g. a wall-smash exposed both at once).
        if next.has_road(stm) {
            let kind = if stm == attacker {
                MoveScoreKind::Win { plies: 1 }
            } else {
                MoveScoreKind::Loss { plies: 1 }
            };
            out.push(MoveScore { mv, kind });
            continue;
        }
        if next.has_road(stm.flip()) {
            // Mover handed a road to their opponent.
            let kind = if stm == attacker {
                MoveScoreKind::Loss { plies: 1 }
            } else {
                MoveScoreKind::Win { plies: 1 }
            };
            out.push(MoveScore { mv, kind });
            continue;
        }

        match next.count_flats() {
            FlatCountOutcome::Win(p) => {
                let outcome = if p == attacker {
                    FlatOutcome::AttackerWin
                } else {
                    FlatOutcome::DefenderWin
                };
                out.push(MoveScore {
                    mv,
                    kind: MoveScoreKind::Flat { outcome },
                });
                continue;
            }
            FlatCountOutcome::Draw => {
                out.push(MoveScore {
                    mv,
                    kind: MoveScoreKind::Flat {
                        outcome: FlatOutcome::Draw,
                    },
                });
                continue;
            }
            FlatCountOutcome::None => {}
        }

        let key = next.key() ^ mask;
        let kind = match tt.probe(key) {
            Some(e) => {
                let v = (e.flags & TT_VALUE_MASK) as u32;
                if (e.flags & TT_FLAG_WIN) != 0 {
                    MoveScoreKind::Win { plies: v + 1 }
                } else {
                    // NoWin entry stores depth searched from the child's
                    // POV; +1 to express the budget that included this move.
                    MoveScoreKind::NoWin { searched: v + 1 }
                }
            }
            None => MoveScoreKind::Unknown,
        };
        out.push(MoveScore { mv, kind });
    }

    out
}

#[inline]
fn neighbors(bb: Bitboard) -> Bitboard {
    bb.shift(Direction::Up)
        | bb.shift(Direction::Down)
        | bb.shift(Direction::Left)
        | bb.shift(Direction::Right)
}

/// Grow `set` through cost-free squares until it stops changing.
fn close_over_zero(mut set: Bitboard, zero: Bitboard) -> Bitboard {
    loop {
        let next = set | (neighbors(set) & zero);
        if next == set {
            return set;
        }
        set = next;
    }
}

/// 0-1 BFS across one axis. `start`/`end` are the two opposing edges.
/// Squares in `zero` are already controlled (free to traverse); every other
/// passable square costs one. `start` is re-injected at each layer so the
/// walk can begin from any edge square, not only one adjacent to the set
/// already reached — the bitboard equivalent of a virtual source node.
fn axis_road_distance(
    start: Bitboard,
    end: Bitboard,
    zero: Bitboard,
    passable: Bitboard,
) -> Option<u32> {
    let mut cur = close_over_zero(start & zero, zero);
    if !(cur & end).is_empty() {
        return Some(0);
    }
    let mut cost = 0u32;
    loop {
        let grown = (neighbors(cur) | start) & passable & !cur;
        if grown.is_empty() {
            return None;
        }
        cost += 1;
        cur = close_over_zero(cur | grown, zero);
        if !(cur & end).is_empty() {
            return Some(cost);
        }
    }
}

/// How many further squares must `player` come to control to complete their
/// nearest road? `None` means no connecting path exists across either axis
/// given the opponent's current blockers.
///
/// Own road pieces cost nothing to traverse, any other non-blocked square
/// costs one, and squares under an opponent wall or capstone are treated as
/// impassable. The result is the cheaper of the two axes.
///
/// # This is a heuristic, not a bound
///
/// Two independent reasons, both of which rule it out of any path where
/// correctness matters:
///
/// 1. It counts squares as if each cost one *placement*, but a single spread
///    can drop stones on several path squares at once. So the true number of
///    attacker moves needed can be lower than this count.
/// 2. Impassability is not permanent. The defender may move a blocking wall
///    or capstone away of their own accord, at which point the square becomes
///    reachable. A `None` therefore means "no road along currently-open
///    lines", not "no road is possible in any continuation".
///
/// Point 2 is worth stating plainly because it is tempting to treat the
/// connectivity test as a free always-safe fast-out. It is not one. Use this
/// only for best-effort sweep marking, gated behind `Limits::prefilter_margin`
/// and never on an explicit single-position solve.
#[must_use]
pub fn road_distance(pos: &Position, player: Player) -> Option<u32> {
    let passable = !pos.blockers(player.flip());
    let zero = pos.roads(player) & passable;

    let vertical = axis_road_distance(
        Bitboard::lower_edge(),
        Bitboard::upper_edge(),
        zero,
        passable,
    );
    let horizontal = axis_road_distance(
        Bitboard::left_edge(),
        Bitboard::right_edge(),
        zero,
        passable,
    );

    match (vertical, horizontal) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (only, None) | (None, only) => only,
    }
}

/// Order attacker moves by likely tinue value. Tiers are separated by
/// large gaps so a move in one tier always beats every move in a worse
/// tier regardless of secondary signals.
///
/// Tiers (lowest rank = tried first):
///   -1000             — immediate road win
///    -10×Δroad_pop    — moves that add ≥1 piece to the road bitboard
///                       (spreads revealing multiple flats outrank flat
///                       placements; placements outrank wall placements
///                       which add 0)
///     100             — spreads that don't grow the road bitboard
///                       (still tactical; can reposition / capture)
///     200             — wall placements
fn order_attacker_moves(pos: &Position, moves: &mut Vec<Move>, attacker: Player) {
    let before_road_pop = pos.roads(attacker).popcount() as i32;
    moves.sort_by_cached_key(|&mv| {
        let after = pos.apply_move(mv);
        if after.has_road(attacker) {
            return -1000i32;
        }
        let delta = after.roads(attacker).popcount() as i32 - before_road_pop;
        if delta > 0 {
            return -10 * delta;
        }
        if mv.is_spread() { 100 } else { 200 }
    });
}

/// Order defender moves by likely defensive value. Same tier structure as
/// the attacker ordering — defender's own road wins first (refutes the
/// tinue outright), then moves that strip attacker road pieces (sliding
/// off a stack, smashing under a cap), then spreads, then placements.
fn order_defender_moves(pos: &Position, moves: &mut Vec<Move>, attacker: Player) {
    let attacker_road_before = pos.roads(attacker).popcount() as i32;
    moves.sort_by_cached_key(|&mv| {
        let after = pos.apply_move(mv);
        if after.has_road(attacker.flip()) {
            return -1000i32;
        }
        let stripped = attacker_road_before - after.roads(attacker).popcount() as i32;
        if stripped > 0 {
            return -10 * stripped;
        }
        if mv.is_spread() { 100 } else { 200 }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::core::SIZE;
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes tests that mutate the process-global board [`SIZE`]. cargo's
    /// default parallel test runner would otherwise let one test's `SIZE.store`
    /// land mid-solve in another, corrupting board geometry and the spread
    /// pattern decode (whose bit positions are relative to the board size).
    /// Poison-tolerant so a panicking assert in one test doesn't cascade into
    /// spurious lock failures elsewhere.
    static SIZE_LOCK: Mutex<()> = Mutex::new(());

    /// Parse a TPS string at `size`, returning the position together with a
    /// guard that pins the global size for the caller's scope. Bind the guard
    /// (`let (pos, _guard) = parse(..)`) so it lives through the whole test.
    #[must_use]
    fn parse(tps: &str, size: u8) -> (Position, MutexGuard<'static, ()>) {
        let guard = SIZE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        SIZE.store(size, Ordering::Release);
        // Reserves live in their own globals and do not follow SIZE.
        crate::board::set_standard_reserves(size);
        let parts: Vec<&str> = tps.split_whitespace().collect();
        (Position::from_tps_parts(&parts).expect("valid tps"), guard)
    }

    fn assert_tinue(tps: &str, size: u8, expected_plies: u32, max_plies: u32) {
        let (pos, _guard) = parse(tps, size);
        let limits = Limits {
            max_plies,
            ..Default::default()
        };
        let (result, _stats) = solve(&pos, &limits);
        match result {
            TinueResult::Tinue { plies, pv, .. } => {
                assert!(
                    plies <= expected_plies,
                    "expected ≤ {} plies, got {} (pv: {:?})",
                    expected_plies,
                    plies,
                    pv
                );
                assert_eq!(plies % 2, 1, "tinue plies must be odd");
            }
            other => panic!("expected tinue, got {:?}", other),
        }
    }

    fn assert_no_tinue(tps: &str, size: u8, max_plies: u32) {
        let (pos, _guard) = parse(tps, size);
        let limits = Limits {
            max_plies,
            ..Default::default()
        };
        let (result, _) = solve(&pos, &limits);
        assert!(
            matches!(result, TinueResult::NoTinue { .. }),
            "expected NoTinue, got {:?}",
            result
        );
    }

    /// Returns the guard alongside the result, and callers must bind it.
    ///
    /// Dropping it at the end of this function would be a trap: `Move`'s
    /// `Display` renders a spread's drop counts relative to
    /// `Position::carry_limit()`, which reads the global `SIZE`. A caller that
    /// released the lock here and only then formatted a move from the PV could
    /// have another test's `SIZE.store` land in between and render the move at
    /// the wrong board size. That produced an intermittent failure in
    /// `full_finds_a_gap_tinue_that_no_tak_chain_reaches` — the verdict was
    /// right, the move string was formatted for the wrong board.
    #[must_use]
    fn solve_scoped(
        tps: &str,
        size: u8,
        scope: TinueScope,
        max_plies: u32,
    ) -> (TinueResult, MutexGuard<'static, ()>) {
        let (pos, guard) = parse(tps, size);
        let limits = Limits {
            max_plies,
            scope,
            ..Default::default()
        };
        (solve(&pos, &limits).0, guard)
    }

    // ---- TinueScope::TakChain -------------------------------------------

    #[test]
    fn tak_chain_finds_the_same_mate_as_full_on_alion_5x5() {
        // Alion's 5x5 mate-in-5 is a strict tak chain, so restricting the
        // move set must not change the verdict or the distance — only the
        // node count (roughly 7x fewer at the time of writing).
        let (result, _guard) = solve_scoped(
            "1,x3,2/2,1C,x2,2/1,1,x2,2/x,1,2C,2,2/x2,1,1,1 2 8",
            5,
            TinueScope::TakChain,
            5,
        );
        match result {
            TinueResult::Tinue { plies, .. } => assert_eq!(plies, 5),
            other => panic!("expected tak-chain tinue in 5, got {:?}", other),
        }
    }

    #[test]
    fn tak_chain_rejects_the_archvenison_gap_tinue() {
        // The counterpart to `archvenison_2026_05_24`: full mode proves a
        // mate in 9, but the winning line opens with a quiet move, so no tak
        // chain reaches it. Topaz reports no_tinue here for the same reason.
        //
        // Cheap despite the depth-9 budget — the restriction collapses the
        // tree to a few hundred nodes — which is why this runs by default
        // while its full-mode twin stays `#[ignore]`d.
        let (result, _guard) = solve_scoped(
            "1,122121S,1,1,1/x,2S,1S,1,1/12,x4/2,2,x,221C,2S/2,2,2,12C,1S 1 24",
            5,
            TinueScope::TakChain,
            9,
        );
        assert!(
            matches!(result, TinueResult::NoTinue { .. }),
            "tak-chain scope must not find the gap tinue, got {:?}",
            result
        );
    }

    #[test]
    fn tak_chain_rejects_the_morten_5s_gap_tinue() {
        // Second known mate-in-9 gap tinue; same expectation as above.
        let (result, _guard) = solve_scoped(
            "2,2221S,2,x2/2,x,2,221S,2/x2,2,x2/12C,2,x,1,x/1221S,1,21121C,1,1 1 28",
            5,
            TinueScope::TakChain,
            9,
        );
        assert!(
            matches!(result, TinueResult::NoTinue { .. }),
            "tak-chain scope must not find the gap tinue, got {:?}",
            result
        );
    }

    #[test]
    fn tak_chain_still_sees_an_immediate_road() {
        // A mate-in-one is a degenerate tak chain. The attacker restriction
        // filters on "leaves a live threat", so a move that *is* the road
        // must be exempted — otherwise the shortest tinues would vanish.
        let (result, _guard) = solve_scoped("x5/x5/x5/x5/1,1,1,1,x 1 5", 5, TinueScope::TakChain, 1);
        match result {
            TinueResult::Tinue { plies, .. } => assert_eq!(plies, 1),
            other => panic!("expected mate in 1, got {:?}", other),
        }
    }

    /// A real mate-in-5 **gap tinue** from a PlayTak game, whose winning first
    /// move `3b3+` threatens nothing — so no tak chain reaches it at any
    /// depth. Sourced from the `topaz_missed_tinues` table of the labelled
    /// puzzle database, i.e. a position independently flagged as one Topaz
    /// could not find.
    ///
    /// This is the shallowest gap tinue on hand, which makes it the fixture
    /// that keeps the scope tests cheap: full mode proves it in ~20k nodes,
    /// while restricted mode refuses it in ~23.
    const GAP_TINUE_5PLY: &str = "2,2,x,x,1/2,2,x,1,x/x,212,1,x,x/x,1,1,x,x/1,x,x,x,x 2 8";

    #[test]
    fn full_finds_a_gap_tinue_that_no_tak_chain_reaches() {
        // The guard must outlive the `to_string()` below; see `solve_scoped`.
        let (full, guard) = solve_scoped(GAP_TINUE_5PLY, 5, TinueScope::Full, 5);
        match full {
            TinueResult::Tinue { plies, pv, .. } => {
                assert_eq!(plies, 5);
                assert_eq!(pv[0].to_string(), "3b3+", "the quiet key move");
            }
            other => panic!("full mode must find this mate in 5, got {:?}", other),
        }
        drop(guard);

        // Searched well past the mate distance to show the restriction is
        // what excludes it, not the depth budget.
        assert!(
            matches!(
                solve_scoped(GAP_TINUE_5PLY, 5, TinueScope::TakChain, 11).0,
                TinueResult::NoTinue { .. }
            ),
            "the winning first move is quiet, so no tak chain can reach this win"
        );
    }

    #[test]
    fn scopes_do_not_share_transposition_entries() {
        // Regression for the TT namespace split, on the case that actually
        // hurts. A TakChain `NoWin` is the weaker claim "no tak-chain win";
        // sharing a key space would let the restricted pass below convince
        // the full pass that this position is quiet — and since it is a gap
        // tinue, full mode is the only thing that can see the win at all.
        //
        // Ordering matters: restricted runs first precisely so its `NoWin`
        // entries are already in the table when full mode probes the same
        // positions.
        let (pos, _guard) = parse(GAP_TINUE_5PLY, 5);
        let mut tt = Tt::new(TT_DEFAULT_BITS);

        let restricted = Limits {
            max_plies: 5,
            scope: TinueScope::TakChain,
            ..Default::default()
        };
        let (early, _) = solve_with_tt(&pos, &restricted, &mut tt);
        assert!(
            matches!(early, TinueResult::NoTinue { .. }),
            "restricted mode cannot see a gap tinue, got {:?}",
            early
        );

        let full = Limits {
            max_plies: 5,
            scope: TinueScope::Full,
            ..Default::default()
        };
        let (result, _) = solve_with_tt(&pos, &full, &mut tt);
        match result {
            TinueResult::Tinue { plies, .. } => assert_eq!(plies, 5),
            other => panic!(
                "full mode must be unaffected by the restricted pass, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn score_moves_does_not_read_across_scopes() {
        // The other half of the namespace split: a full-scope solve's
        // verdicts must be invisible to a TakChain lookup, so callers that
        // pass a mismatched scope get `Unknown` rather than another scope's
        // answers.
        let (pos, _guard) = parse("x5/x5/x5/x5/1,1,1,1,x 1 5", 5);
        let limits = Limits {
            max_plies: 1,
            ..Default::default()
        };
        let mut tt = Tt::new(TT_DEFAULT_BITS);
        let (_, _) = solve_with_tt(&pos, &limits, &mut tt);

        // a2 is a quiet placement: not an immediate road either way, so its
        // verdict can only come from the TT — which makes it a clean probe
        // for whether the namespaces leak.
        let matching = score_moves(&pos, Player::P1, TinueScope::Full, &tt);
        let mismatched = score_moves(&pos, Player::P1, TinueScope::TakChain, &tt);
        for scores in [&matching, &mismatched] {
            let a2 = scores.iter().find(|s| s.mv.to_string() == "a2").unwrap();
            assert!(
                matches!(a2.kind, MoveScoreKind::Unknown),
                "a2 was never searched at depth 1, expected Unknown, got {:?}",
                a2.kind
            );
        }
    }

    // ---- road_distance / sweep pre-filter --------------------------------

    #[test]
    fn road_distance_counts_squares_still_needed() {
        // Empty 5x5: five squares to cross, none of them owned yet.
        let (pos, _guard) = parse("x5/x5/x5/x5/x5 1 3", 5);
        assert_eq!(road_distance(&pos, Player::P1), Some(5));
        drop(_guard);

        // Four flats on rank 1 — one square short of a road.
        let (pos, _guard) = parse("x5/x5/x5/x5/1,1,1,1,x 1 5", 5);
        assert_eq!(road_distance(&pos, Player::P1), Some(1));
        drop(_guard);

        // A finished road costs nothing further.
        let (pos, _guard) = parse("x5/x5/x5/x5/1,1,1,1,1 2 6", 5);
        assert_eq!(road_distance(&pos, Player::P1), Some(0));
    }

    #[test]
    fn road_distance_is_none_when_every_line_is_walled_off() {
        // P2 walls fill rank 3 and file c, so no unblocked path crosses
        // either axis for P1 given the current blockers.
        let (pos, _guard) = parse("x2,2S,x2/x2,2S,x2/2S,2S,2S,2S,2S/x2,2S,x2/x2,2S,x2 1 10", 5);
        assert_eq!(road_distance(&pos, Player::P1), None);
    }

    #[test]
    fn prefilter_skips_only_far_positions() {
        // A mate-in-one must never be skipped, however tight the margin.
        let (pos, _guard) = parse("x5/x5/x5/x5/1,1,1,1,x 1 5", 5);
        assert!(!prefilter_skips(&pos, 5, 0));
        drop(_guard);

        // An empty board needs 5 squares; a 3-ply horizon buys the attacker
        // 2 moves, so with no margin it is skipped and with a margin of 3 it
        // is not.
        let (pos, _guard) = parse("x5/x5/x5/x5/x5 1 3", 5);
        assert!(prefilter_skips(&pos, 3, 0));
        assert!(!prefilter_skips(&pos, 3, 3));
    }

    #[test]
    fn prefilter_is_off_by_default() {
        // The filter is heuristic, so nothing may enable it implicitly.
        assert_eq!(Limits::default().prefilter_margin, None);
        assert_eq!(Limits::default().scope, TinueScope::Full);
    }

    #[test]
    fn mate_in_one_5x5() {
        // P1 has 4 flats on rank 1; placing on e1 (any piece) completes the road.
        assert_tinue("x5/x5/x5/x5/1,1,1,1,x 1 5", 5, 1, 1);
    }

    #[test]
    fn mate_in_one_6x6() {
        assert_tinue("x6/x6/x6/x6/x6/1,1,1,1,1,x 1 6", 6, 1, 1);
    }

    #[test]
    fn mate_in_one_7x7() {
        assert_tinue("x7/x7/x7/x7/x7/x7/1,1,1,1,1,1,x 1 7", 7, 1, 1);
    }

    #[test]
    fn alion_5x5_tinue() {
        // 5x5 Tinuë Pattern.ptn — P2 to move, mate-in-5
        assert_tinue(
            "1,x3,2/2,1C,x2,2/1,1,x2,2/x,1,2C,2,2/x2,1,1,1 2 8",
            5,
            5,
            5,
        );
    }

    #[test]
    fn alion_6x6_puzzle1() {
        // Alion's Puzzle #1 (Tinuë).ptn — P2 to move, mate-in-7
        assert_tinue(
            "2,1221122,1,1,1,2S/1,1,1,x,1C,1111212/x2,2,212,2C,11/2,2,x2,1,1/x3,1,1,x/x2,2,21,x,112S 2 32",
            6,
            7,
            7,
        );
    }

    #[test]
    fn alion_6x6_puzzle2() {
        // Alion's Puzzle #2 (Tinuë).ptn — P1 to move, mate-in-7
        assert_tinue(
            "2,212221C,2,2,2C,1/1,2,1,1,2,1/12,x,1S,2S,2,1/2,2,2,x2,1/1,2212121S,2,12,1,1S/x,2,2,2,x,1 1 30",
            6,
            7,
            7,
        );
    }

    #[test]
    fn empty_5x5_no_tinue() {
        // Almost-empty board: nobody is close to a road.
        assert_no_tinue("x5/x5/x5/x5/2,1,x3 1 2", 5, 5);
    }

    // Known-hard puzzles — ignored by default. Currently exceed practical
    // wall time even with TT; will be revisited once defender-side move
    // pruning (restrict to road-relevant moves) and threat extensions land.
    // Run with: cargo test --release -- --ignored tinue::tests
    #[test]
    #[ignore]
    fn alion_6x6_puzzle3() {
        // Alion's Puzzle #3 (Tinuë).ptn — P1 to move
        assert_tinue(
            "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,2,x3,22S 1 27",
            6,
            11,
            11,
        );
    }

    #[test]
    #[ignore]
    fn morten_5s_tinue_2() {
        // Morten 5s tinue #2.ptn — P1 to move
        assert_tinue(
            "2,2221S,2,x2/2,x,2,221S,2/x2,2,x2/12C,2,x,1,x/1221S,1,21121C,1,1 1 28",
            5,
            11,
            11,
        );
    }

    #[test]
    #[ignore]
    fn gruppler_2025_03_10_puzzle1() {
        // Gruppler's Tinue Puzzle 2025-03-10 #1.ptn — P1 to move
        assert_tinue(
            "x4,2,112/1,1,1,1,1212C,1121C/x,21,x,2,121,12/2,1,1,21,x2/x,2,2,x,1,2/2,2,x3,2 1 28",
            6,
            11,
            11,
        );
    }

    #[test]
    #[ignore]
    fn archvenison_2026_05_24() {
        // "archvenison 24-05-26" (PlayTak, 2026-05-24) — P1 to move.
        // A *non-tak-chain* (gap) tinue: the forced win passes through a quiet
        // non-threatening move, so a tak-chain-only solver (e.g. Topaz) reports
        // no_tinue. syntaks's full-width search finds it. Confirmed mate-in-9
        // (unassisted solve: 8.5M nodes, ~8 min).
        assert_tinue(
            "1,122121S,1,1,1/x,2S,1S,1,1/12,x4/2,2,x,221C,2S/2,2,2,12C,1S 1 24",
            5,
            9,
            9,
        );
    }

    #[test]
    fn score_moves_marks_winning_first_move() {
        // Mate-in-one: P1 places on e1 to complete a rank-1 road. After
        // solving, score_moves should report `Win { plies: 1 }` for that
        // move and report the other empty squares as non-winning placements.
        let (pos, _guard) = parse("x5/x5/x5/x5/1,1,1,1,x 1 5", 5);
        let limits = Limits {
            max_plies: 1,
            ..Default::default()
        };
        let mut tt = Tt::new(TT_DEFAULT_BITS);
        let (result, _) = solve_with_tt(&pos, &limits, &mut tt);
        assert!(matches!(result, TinueResult::Tinue { plies: 1, .. }));

        let scores = score_moves(&pos, Player::P1, TinueScope::Full, &tt);
        let win = scores
            .iter()
            .find(|s| s.mv.to_string() == "e1")
            .expect("e1 in legal moves");
        assert!(
            matches!(win.kind, MoveScoreKind::Win { plies: 1 }),
            "expected Win {{ plies: 1 }} for e1, got {:?}",
            win.kind
        );
        // A flat placement on an unrelated square (e.g. a2) doesn't win
        // immediately, and the post-position wasn't visited by the
        // depth-1 search either — should be Unknown.
        let other = scores
            .iter()
            .find(|s| s.mv.to_string() == "a2")
            .expect("a2 in legal moves");
        assert!(
            matches!(other.kind, MoveScoreKind::Unknown),
            "expected Unknown for a2, got {:?}",
            other.kind
        );
    }

    #[test]
    fn score_moves_marks_all_root_winners_after_solve() {
        // After solving Alion's 5x5 mate-in-5 with find_all_winners on,
        // every reported winning_first_move should also show up in
        // score_moves as a `Win` entry — they share the same TT lookup
        // path, so this guards against the JSON shape diverging from the
        // TT semantics.
        let (pos, _guard) = parse(
            "1,x3,2/2,1C,x2,2/1,1,x2,2/x,1,2C,2,2/x2,1,1,1 2 8",
            5,
        );
        let limits = Limits {
            max_plies: 5,
            ..Default::default()
        };
        let mut tt = Tt::new(TT_DEFAULT_BITS);
        let (result, _) = solve_with_tt(&pos, &limits, &mut tt);
        let winners = match result {
            TinueResult::Tinue {
                winning_first_moves,
                ..
            } => winning_first_moves,
            other => panic!("expected tinue, got {:?}", other),
        };
        assert!(!winners.is_empty());

        let scores = score_moves(&pos, Player::P2, TinueScope::Full, &tt);
        for w in &winners {
            let entry = scores
                .iter()
                .find(|s| s.mv == *w)
                .unwrap_or_else(|| panic!("winner {} missing from score_moves", w));
            assert!(
                matches!(entry.kind, MoveScoreKind::Win { .. }),
                "winner {} scored as {:?}, expected Win",
                w,
                entry.kind
            );
        }
    }

    #[test]
    fn parse_spread_to_board_edge() {
        // Regression: parsing "3c3-12" used to be rejected as illegal because
        // the parser added a phantom advance bit at the cumulative drop
        // position, overcounting count_ones() by 1 — which made is_legal
        // refuse spreads reaching the exact board edge.
        let (pos, _guard) = parse(
            "x2,1,21,2,2/1,2,21,1,21,2/1S,2,2,2C,2,2/21S,1,121C,x,1,12/2,2,121,1,1,1/2,x,12,x2,22S 1 28",
            6,
        );
        let mv: Move = "3c3-12".parse().expect("parse");
        assert!(pos.is_legal(mv), "3c3-12 should be legal here");
    }
}

/// Solve for a tinue at `pos`. Allocates a fresh TT internally.
pub fn solve<'a>(pos: &Position, limits: &Limits<'a>) -> (TinueResult, Stats) {
    let mut tt = Tt::new(TT_DEFAULT_BITS);
    solve_with_tt(pos, limits, &mut tt)
}

/// Run one root-level search at `depth`, restricting the attacker's *first*
/// move — and only the first — to `root_move`. This answers "does this
/// specific candidate force a win at `depth`?" without wading through (and
/// deep-searching) every other root move first. Everything below the root is
/// the normal search for the configured [`TinueScope`], so the defender still
/// gets every reply the scope allows and the proof stays sound. If
/// `root_move` wins, the position *is* tinue — one winning attacker move
/// suffices — and `plies` is that move's mate length.
///
/// Use this for candidate verification or deep analysis of a parked position
/// where a strong first move is already suspected: notably a quiet non-tak
/// move, which the tak-chain-biased move ordering would otherwise try last.
///
/// Under [`TinueScope::TakChain`] a `root_move` that leaves no live threat is
/// reported `NoTinue` immediately — it is not a legal link in a tak chain, so
/// there is nothing to search.
pub fn solve_one_depth_root_move<'a>(
    pos: &Position,
    depth: u32,
    tt: &mut Tt,
    limits: &Limits<'a>,
    root_move: Move,
) -> (TinueResult, Stats) {
    let attacker = pos.stm();

    // Terminal short-circuit, identical to solve_one_depth.
    if pos.has_road(attacker) || pos.has_road(attacker.flip()) {
        return (
            TinueResult::NoTinue { searched_plies: 0 },
            Stats::default(),
        );
    }

    let mut searcher = Searcher::new(attacker, limits, tt);
    let mut pv: Vec<Move> = Vec::with_capacity(depth as usize);

    // Single-move root expansion: mirror one iteration of search_attacker's
    // per-move loop for `root_move` only.
    let outcome = 'root: {
        let next = pos.apply_move(root_move);
        if next.has_road(attacker) {
            pv.push(root_move);
            break 'root NodeOutcome::AttackerWins(1);
        }
        // The candidate handed the defender a road, or reached a flat-count
        // terminal — either way it is not a winning first move.
        if next.has_road(attacker.flip()) || !matches!(next.count_flats(), FlatCountOutcome::None) {
            break 'root NodeOutcome::DefenderHolds;
        }
        if searcher.restricted() && !searcher.attacker_threatens_road(&next) {
            break 'root NodeOutcome::DefenderHolds;
        }
        let mut sub_pv = Vec::new();
        match searcher.search_defender(&next, depth - 1, &mut sub_pv) {
            NodeOutcome::AttackerWins(plies) => {
                pv.push(root_move);
                pv.extend_from_slice(&sub_pv);
                NodeOutcome::AttackerWins(plies + 1)
            }
            other => other,
        }
    };

    let nodes = searcher.nodes.load(Ordering::Relaxed);
    match outcome {
        NodeOutcome::AttackerWins(plies) => {
            pv.truncate(plies as usize);
            let winners = if pv.is_empty() {
                Vec::new()
            } else {
                vec![pv[0]]
            };
            (
                TinueResult::Tinue {
                    plies,
                    pv,
                    winning_first_moves: winners,
                },
                Stats {
                    nodes,
                    max_depth_reached: depth,
                },
            )
        }
        NodeOutcome::DefenderHolds => (
            TinueResult::NoTinue {
                searched_plies: depth,
            },
            Stats {
                nodes,
                max_depth_reached: depth,
            },
        ),
        NodeOutcome::Aborted => {
            let reason = if nodes >= limits.max_nodes {
                AbortReason::Nodes
            } else {
                AbortReason::Cancelled
            };
            (
                TinueResult::Aborted {
                    reason,
                    searched_plies: 0,
                },
                Stats {
                    nodes,
                    max_depth_reached: depth,
                },
            )
        }
    }
}

/// Run exactly one root-level search at the given odd depth. Use this when
/// you need to drive iterative deepening from outside (e.g., the wasm
/// streaming path that posts a progress event per completed depth). The
/// caller-supplied `tt` is shared across calls, so successive depths get
/// the same warm-cache benefit as the internal iterative-deepening loop
/// in [`solve_with_tt`].
///
/// Node budget in `limits.max_nodes` applies per-call (not cumulative),
/// matching its semantics in [`solve_with_tt`] for a single depth.
pub fn solve_one_depth<'a>(
    pos: &Position,
    depth: u32,
    tt: &mut Tt,
    limits: &Limits<'a>,
) -> (TinueResult, Stats) {
    let attacker = pos.stm();

    // Terminal positions short-circuit identically to solve_with_tt.
    if pos.has_road(attacker) || pos.has_road(attacker.flip()) {
        return (
            TinueResult::NoTinue { searched_plies: 0 },
            Stats::default(),
        );
    }

    let mut searcher = Searcher::new(attacker, limits, tt);
    let mut pv = Vec::with_capacity(depth as usize);
    let outcome = searcher.search_attacker(pos, depth, &mut pv);
    let nodes = searcher.nodes.load(Ordering::Relaxed);

    match outcome {
        NodeOutcome::AttackerWins(plies) => {
            searcher.extend_pv_via_tt(pos, &mut pv, plies as usize);
            pv.truncate(plies as usize);
            let mut winners = if pv.is_empty() {
                Vec::new()
            } else {
                vec![pv[0]]
            };
            if limits.find_all_winners && !pv.is_empty() {
                searcher.collect_root_winners(pos, depth, pv[0], &mut winners);
            }
            (
                TinueResult::Tinue {
                    plies,
                    pv,
                    winning_first_moves: winners,
                },
                Stats {
                    nodes,
                    max_depth_reached: depth,
                },
            )
        }
        NodeOutcome::DefenderHolds => (
            TinueResult::NoTinue {
                searched_plies: depth,
            },
            Stats {
                nodes,
                max_depth_reached: depth,
            },
        ),
        NodeOutcome::Aborted => {
            let reason = if nodes >= limits.max_nodes {
                AbortReason::Nodes
            } else {
                AbortReason::Cancelled
            };
            (
                TinueResult::Aborted {
                    reason,
                    searched_plies: 0,
                },
                Stats {
                    nodes,
                    max_depth_reached: depth,
                },
            )
        }
    }
}

/// Solve for a tinue at `pos` reusing the caller's TT. Pass the same `tt` to
/// successive calls (e.g. when sweeping a game) to share cached results.
pub fn solve_with_tt<'a>(
    pos: &Position,
    limits: &Limits<'a>,
    tt: &mut Tt,
) -> (TinueResult, Stats) {
    let attacker = pos.stm();

    // Terminal positions aren't "tinue" candidates — if a road already
    // exists on the board (for either side), the game is over and there's
    // no forced sequence to find. Reporting Tinue here would be wrong (the
    // ply that finished the road would otherwise get a spurious tak/tinue
    // mark when callers query the post-move TPS).
    if pos.has_road(attacker) || pos.has_road(attacker.flip()) {
        return (
            TinueResult::NoTinue { searched_plies: 0 },
            Stats::default(),
        );
    }

    // Best-effort sweep skip; disabled by default. See `prefilter_margin`.
    if limits
        .prefilter_margin
        .is_some_and(|margin| prefilter_skips(pos, limits.max_plies, margin))
    {
        return (
            TinueResult::NoTinue {
                searched_plies: limits.max_plies,
            },
            Stats::default(),
        );
    }

    let mut searcher = Searcher::new(attacker, limits, tt);

    let mut last_searched = 0u32;
    let mut depth = 1u32;

    while depth <= limits.max_plies {
        let mut pv = Vec::with_capacity(depth as usize);
        let outcome = searcher.search_attacker(pos, depth, &mut pv);

        last_searched = depth;

        match outcome {
            NodeOutcome::AttackerWins(plies) => {
                // TT cutoffs can truncate the PV — extend by walking the
                // chain of stored best_moves until we reach `plies` length
                // or run out of entries.
                searcher.extend_pv_via_tt(pos, &mut pv, plies as usize);
                pv.truncate(plies as usize);

                // Enumerate alternate root winners at the same depth so
                // callers can mark every move that's on the road to tinue,
                // not just the primary PV's first ply. Cheap: the TT is
                // hot from the primary search, so non-winning moves get
                // cut fast and any other proven winners are TT hits.
                let mut winners = if pv.is_empty() {
                    Vec::new()
                } else {
                    vec![pv[0]]
                };
                if limits.find_all_winners && !pv.is_empty() {
                    searcher.collect_root_winners(pos, depth, pv[0], &mut winners);
                }

                let stats = Stats {
                    nodes: searcher.nodes.load(Ordering::Relaxed),
                    max_depth_reached: depth,
                };
                return (
                    TinueResult::Tinue {
                        plies,
                        pv,
                        winning_first_moves: winners,
                    },
                    stats,
                );
            }
            NodeOutcome::DefenderHolds => {}
            NodeOutcome::Aborted => {
                let stats = Stats {
                    nodes: searcher.nodes.load(Ordering::Relaxed),
                    max_depth_reached: last_searched,
                };
                let reason =
                    if searcher.nodes.load(Ordering::Relaxed) >= limits.max_nodes {
                        AbortReason::Nodes
                    } else {
                        AbortReason::Cancelled
                    };
                return (
                    TinueResult::Aborted {
                        reason,
                        searched_plies: last_searched,
                    },
                    stats,
                );
            }
        }

        depth += 2;
    }

    let stats = Stats {
        nodes: searcher.nodes.load(Ordering::Relaxed),
        max_depth_reached: last_searched,
    };
    (
        TinueResult::NoTinue {
            searched_plies: last_searched,
        },
        stats,
    )
}
