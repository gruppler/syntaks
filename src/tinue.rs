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

use crate::board::{FlatCountOutcome, Position};
use crate::core::Player;
use crate::movegen::generate_moves;
use crate::takmove::Move;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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

struct Searcher<'a> {
    attacker: Player,
    nodes: AtomicU64,
    node_limit: u64,
    cancel: Option<&'a AtomicBool>,
    aborted: bool,
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

impl<'a> Searcher<'a> {
    fn new(attacker: Player, limits: &Limits<'a>) -> Self {
        Self {
            attacker,
            nodes: AtomicU64::new(0),
            node_limit: limits.max_nodes,
            cancel: limits.cancel,
            aborted: false,
        }
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

        self.bump_nodes();

        let mut moves = Vec::with_capacity(64);
        generate_moves(&mut moves, pos);
        order_attacker_moves(pos, &mut moves, self.attacker);

        for &mv in &moves {
            let next = pos.apply_move(mv);

            // After attacker's move: if attacker has a road, that's a win.
            // Per current-player-wins rule, even if defender also has a road
            // on the resulting position, attacker (the mover) wins.
            if next.has_road(self.attacker) {
                pv.clear();
                pv.push(mv);
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
                    return NodeOutcome::AttackerWins(plies + 1);
                }
                NodeOutcome::DefenderHolds => {}
                NodeOutcome::Aborted => return NodeOutcome::Aborted,
            }
        }

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

        self.bump_nodes();

        let mut moves = Vec::with_capacity(64);
        generate_moves(&mut moves, pos);

        if moves.is_empty() {
            return NodeOutcome::DefenderHolds;
        }

        order_defender_moves(pos, &mut moves, self.attacker);

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
                    return NodeOutcome::DefenderHolds;
                }
                FlatCountOutcome::Draw => {
                    // a draw refutes tinue
                    pv.clear();
                    pv.push(mv);
                    return NodeOutcome::DefenderHolds;
                }
                FlatCountOutcome::Win(_) => {
                    // attacker wins flat count — but tinue requires a road
                    // win, so this is also a refutation.
                    pv.clear();
                    pv.push(mv);
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
                    return NodeOutcome::DefenderHolds;
                }
                NodeOutcome::Aborted => return NodeOutcome::Aborted,
            }
        }

        match longest {
            Some((plies, found_pv)) => {
                *pv = found_pv;
                NodeOutcome::AttackerWins(plies)
            }
            None => NodeOutcome::DefenderHolds,
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

/// Solve for a tinue at `pos` from the side-to-move's perspective. Iteratively
/// deepens over odd ply counts (1, 3, 5, ...) up to `limits.max_plies`.
pub fn solve<'a>(pos: &Position, limits: &Limits<'a>) -> (TinueResult, Stats) {
    let attacker = pos.stm();
    let mut searcher = Searcher::new(attacker, limits);

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
