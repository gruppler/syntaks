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
use crate::core::{Direction, Player};
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
pub struct Tt {
    entries: Vec<TtEntry>,
    mask: usize,
}

impl Tt {
    pub fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        Self {
            entries: vec![TtEntry::default(); size],
            mask: size - 1,
        }
    }

    pub fn clear(&mut self) {
        for e in &mut self.entries {
            *e = TtEntry::default();
        }
    }

    fn idx(&self, key: u64) -> usize {
        (key as usize) & self.mask
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

#[inline]
fn attacker_key_mask(attacker: Player) -> u64 {
    match attacker {
        Player::P1 => ATTACKER_KEY_MASK_P1,
        Player::P2 => ATTACKER_KEY_MASK_P2,
    }
}

/// Result of a tinue search.
#[derive(Clone, Debug)]
pub enum TinueResult {
    /// Forced road win in `plies` ply (always odd: attacker plays the last
    /// move). `pv` is one principal variation; non-PV defender moves also
    /// lose, but only the longest defense's continuation is recorded.
    Tinue { plies: u32, pv: Vec<Move> },

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

#[derive(Clone)]
pub struct Limits<'a> {
    pub max_plies: u32,
    pub max_nodes: u64,
    pub cancel: Option<&'a AtomicBool>,
}

impl Default for Limits<'_> {
    fn default() -> Self {
        Self {
            max_plies: 21,
            max_nodes: u64::MAX,
            cancel: None,
        }
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub nodes: u64,
    pub max_depth_reached: u32,
}

struct Searcher<'a, 'b> {
    attacker: Player,
    /// XOR'd into every TT key so entries are partitioned by attacker.
    /// See [`attacker_key_mask`] for the rationale.
    attacker_mask: u64,
    nodes: AtomicU64,
    node_limit: u64,
    cancel: Option<&'a AtomicBool>,
    aborted: bool,
    tt: &'b mut Tt,
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
            attacker_mask: attacker_key_mask(attacker),
            nodes: AtomicU64::new(0),
            node_limit: limits.max_nodes,
            cancel: limits.cancel,
            aborted: false,
            tt,
        }
    }

    #[inline]
    fn tt_key(&self, pos: &Position) -> u64 {
        pos.key() ^ self.attacker_mask
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

        // Defender-move pruning is intentionally NOT applied here. An
        // earlier "road-relevance zone" filter (kept the move only if its
        // source/target was in or one orthogonal step from a road piece)
        // turned out to be unsound: it ignored intermediate drop squares of
        // a spread, so a spread starting and ending outside the zone but
        // dropping a stone on a critical road-blocking square was wrongly
        // pruned. That produced false-positive Tinuës in real games.
        // Soundness > speed for tinue annotation.

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

/// Order attacker moves so that road-completing moves come first, then
/// moves that create new road threats, then everything else. Cheap and
/// drastically improves pruning.
fn order_attacker_moves(pos: &Position, moves: &mut Vec<Move>, attacker: Player) {
    moves.sort_by_cached_key(|&mv| {
        let after = pos.apply_move(mv);
        // 0: road win
        if after.has_road(attacker) {
            return 0i32;
        }
        // 1: a move that brings attacker closer to road completion (more
        //    pieces in the road bitboard than before is a coarse proxy).
        let before_road_pop = pos.roads(attacker).popcount() as i32;
        let after_road_pop = after.roads(attacker).popcount() as i32;
        if after_road_pop > before_road_pop {
            return 1 - (after_road_pop - before_road_pop);
        }
        // 2: spreads (typically tactical) before placements.
        if mv.is_spread() { 100 } else { 200 }
    });
}

/// Order defender moves by likely defensive value: moves that drop the
/// attacker's road bitboard population, blocks adjacent to attacker road
/// pieces, and walls/caps before flats.
fn order_defender_moves(pos: &Position, moves: &mut Vec<Move>, attacker: Player) {
    let attacker_road_before = pos.roads(attacker).popcount() as i32;
    moves.sort_by_cached_key(|&mv| {
        let after = pos.apply_move(mv);
        // 0: defender wins (own road)
        if after.has_road(attacker.flip()) {
            return 0i32;
        }
        // 1: reduces attacker's road population (block / capture)
        let attacker_road_after = after.roads(attacker).popcount() as i32;
        if attacker_road_after < attacker_road_before {
            return 10 - (attacker_road_before - attacker_road_after);
        }
        // 2: spreads before placements
        if mv.is_spread() { 100 } else { 200 }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Position;
    use crate::core::SIZE;
    use std::sync::atomic::Ordering;

    fn parse(tps: &str, size: u8) -> Position {
        SIZE.store(size, Ordering::Release);
        let parts: Vec<&str> = tps.split_whitespace().collect();
        Position::from_tps_parts(&parts).expect("valid tps")
    }

    fn assert_tinue(tps: &str, size: u8, expected_plies: u32, max_plies: u32) {
        let pos = parse(tps, size);
        let limits = Limits {
            max_plies,
            ..Default::default()
        };
        let (result, _stats) = solve(&pos, &limits);
        match result {
            TinueResult::Tinue { plies, pv } => {
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
        let pos = parse(tps, size);
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
    fn parse_spread_to_board_edge() {
        // Regression: parsing "3c3-12" used to be rejected as illegal because
        // the parser added a phantom advance bit at the cumulative drop
        // position, overcounting count_ones() by 1 — which made is_legal
        // refuse spreads reaching the exact board edge.
        let pos = parse(
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

/// Solve for a tinue at `pos` reusing the caller's TT. Pass the same `tt` to
/// successive calls (e.g. when sweeping a game) to share cached results.
pub fn solve_with_tt<'a>(
    pos: &Position,
    limits: &Limits<'a>,
    tt: &mut Tt,
) -> (TinueResult, Stats) {
    let attacker = pos.stm();
    let mut searcher = Searcher::new(attacker, limits, tt);

    let mut last_searched = 0u32;
    let mut depth = 1u32;

    while depth <= limits.max_plies {
        let mut pv = Vec::with_capacity(depth as usize);
        let outcome = searcher.search_attacker(pos, depth, &mut pv);

        last_searched = depth;

        match outcome {
            NodeOutcome::AttackerWins(plies) => {
                let stats = Stats {
                    nodes: searcher.nodes.load(Ordering::Relaxed),
                    max_depth_reached: depth,
                };
                // TT cutoffs can truncate the PV — extend by walking the
                // chain of stored best_moves until we reach `plies` length
                // or run out of entries.
                searcher.extend_pv_via_tt(pos, &mut pv, plies as usize);
                pv.truncate(plies as usize);
                return (TinueResult::Tinue { plies, pv }, stats);
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
