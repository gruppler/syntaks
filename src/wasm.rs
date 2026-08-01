/*
 * syntaks, a TEI Tak engine — wasm bindings.
 */

use crate::board::{set_standard_reserves, Position};
use crate::core::{Player, SIZE};
use crate::tinue::{
    self, AbortReason, FlatOutcome, Limits, MoveScoreKind, TinueResult, TinueScope, Tt,
};
use serde::Serialize;
use serde_wasm_bindgen::Serializer;
use std::sync::atomic::Ordering;
use wasm_bindgen::prelude::*;

// serde_wasm_bindgen defaults to emitting JS `Map` for serde struct/map
// types — including any struct that uses `#[serde(flatten)]`, which
// forces the map serializer because the field set isn't static at
// compile time. We want plain JS objects so consumers can read
// `entry.move` / `entry.kind` directly.
fn to_js<T: Serialize>(value: &T) -> JsValue {
    let ser = Serializer::new().serialize_maps_as_objects(true);
    value.serialize(&ser).unwrap_or(JsValue::NULL)
}

#[derive(Serialize)]
#[serde(tag = "kind")]
enum SolveOutcome {
    #[serde(rename = "tinue")]
    Tinue {
        plies: u32,
        pv: Vec<String>,
        /// Every attacker move at the root that wins at `plies` depth.
        /// Always contains `pv[0]`. Lets callers mark every winning move
        /// played in any branch, not only the engine's preferred PV.
        winning_first_moves: Vec<String>,
    },
    #[serde(rename = "no_tinue")]
    NoTinue { searched_plies: u32 },
    #[serde(rename = "aborted")]
    Aborted {
        reason: &'static str,
        searched_plies: u32,
    },
    #[serde(rename = "error")]
    Error { message: String },
}

#[derive(Serialize)]
struct SolveResponse {
    outcome: SolveOutcome,
    nodes: u64,
}

fn parse_max_nodes(max_nodes: f64) -> u64 {
    if max_nodes <= 0.0 || !max_nodes.is_finite() {
        u64::MAX
    } else {
        max_nodes.min(u64::MAX as f64) as u64
    }
}

fn build_response(result: TinueResult, stats: tinue::Stats) -> SolveResponse {
    let outcome = match result {
        TinueResult::Tinue {
            plies,
            pv,
            winning_first_moves,
        } => SolveOutcome::Tinue {
            plies,
            pv: pv.iter().map(|m| m.to_string()).collect(),
            winning_first_moves: winning_first_moves
                .iter()
                .map(|m| m.to_string())
                .collect(),
        },
        TinueResult::NoTinue { searched_plies } => SolveOutcome::NoTinue { searched_plies },
        TinueResult::Aborted {
            reason,
            searched_plies,
        } => SolveOutcome::Aborted {
            reason: match reason {
                AbortReason::Cancelled => "cancelled",
                AbortReason::Nodes => "nodes",
                AbortReason::Depth => "depth",
            },
            searched_plies,
        },
    };
    SolveResponse {
        outcome,
        nodes: stats.nodes,
    }
}

/// Per-legal-move verdict surfaced to JS by [`TinueSolver::score_moves`].
/// Mirrors [`MoveScoreKind`] but flattened into `kind`-tagged JSON for
/// direct UI consumption.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum MoveScoreEntryKind {
    Win { plies: u32 },
    Loss { plies: u32 },
    NoWin { searched: u32 },
    Flat { outcome: &'static str },
    Unknown,
}

#[derive(Serialize)]
struct MoveScoreEntry {
    #[serde(rename = "move")]
    mv: String,
    #[serde(flatten)]
    kind: MoveScoreEntryKind,
}

fn flat_outcome_str(outcome: FlatOutcome) -> &'static str {
    match outcome {
        FlatOutcome::AttackerWin => "win",
        FlatOutcome::DefenderWin => "loss",
        FlatOutcome::Draw => "draw",
    }
}

/// Map the JS `scope` argument onto [`TinueScope`]. `undefined`/`null` and
/// the empty string mean `full`, so callers written before the scope toggle
/// existed keep the complete (gap-tinue-finding) semantics they had.
fn parse_scope(scope: Option<String>) -> Result<TinueScope, String> {
    match scope.as_deref().map(str::trim) {
        None | Some("") => Ok(TinueScope::Full),
        Some(s) => s
            .parse()
            .map_err(|_| format!("unknown scope {:?} (expected \"full\" or \"tak-chain\")", s)),
    }
}

/// A negative or non-finite margin disables the sweep pre-filter, matching
/// how `max_nodes` treats out-of-range values.
fn parse_prefilter_margin(margin: f64) -> Option<u32> {
    if margin < 0.0 || !margin.is_finite() {
        None
    } else {
        Some(margin.min(u32::MAX as f64) as u32)
    }
}

fn parse_position(tps: &str, size: u8) -> Result<Position, String> {
    if !(5..=7).contains(&size) {
        return Err(format!("unsupported size {} (only 5/6/7)", size));
    }
    SIZE.store(size, Ordering::Release);
    // Reserves are a separate global and do not follow SIZE; without this a
    // 5x5 solve runs with 30 flats per player instead of 21.
    set_standard_reserves(size);
    let parts: Vec<&str> = tps.split_whitespace().collect();
    Position::from_tps_parts(&parts).map_err(|e| format!("tps parse: {:?}", e))
}

fn to_jsvalue(response: SolveResponse) -> JsValue {
    to_js(&response)
}

fn error_response(message: String) -> JsValue {
    to_jsvalue(SolveResponse {
        outcome: SolveOutcome::Error { message },
        nodes: 0,
    })
}

/// One-shot tinue solve with a fresh internal TT. Use [`TinueSolver`] to share
/// a TT across calls (sweep mode).
///
/// `max_plies` caps iterative deepening; `max_nodes` is a node budget (0 / NaN
/// / negative = no cap). Returns `{ outcome: { kind, ... }, nodes }`.
///
/// `scope` is `"full"` (default when omitted) or `"tak-chain"`. Under
/// `"tak-chain"` a `no_tinue` outcome means **no tak-chain tinue** — a gap
/// tinue may still exist — so the UI must label it as such rather than
/// claiming the position is not tinue.
#[wasm_bindgen]
pub fn solve_tinue(
    tps: &str,
    size: u8,
    max_plies: u32,
    max_nodes: f64,
    scope: Option<String>,
) -> JsValue {
    let pos = match parse_position(tps, size) {
        Ok(p) => p,
        Err(message) => return error_response(message),
    };
    let scope = match parse_scope(scope) {
        Ok(s) => s,
        Err(message) => return error_response(message),
    };
    let limits = Limits {
        max_plies,
        max_nodes: parse_max_nodes(max_nodes),
        scope,
        ..Default::default()
    };
    let (result, stats) = tinue::solve(&pos, &limits);
    to_jsvalue(build_response(result, stats))
}

/// Stateful tinue solver that retains its transposition table across calls.
/// Constructed with a `bits` parameter sizing the TT (entries = 1 << bits,
/// 16 B each — e.g. bits=22 → 64 MB). Use this when sweeping a game so each
/// position's search seeds the next.
#[wasm_bindgen]
pub struct TinueSolver {
    tt: Tt,
}

#[wasm_bindgen]
impl TinueSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(bits: u32) -> TinueSolver {
        // Build the hits magic table now rather than during the first
        // search. On wasm it's behind a `OnceLock` and would otherwise
        // stall the initial query by tens of ms.
        crate::hits::preload();

        let bits = bits.clamp(10, 28);
        TinueSolver {
            tt: Tt::new(bits),
        }
    }

    /// Wipe the cached entries without reallocating.
    pub fn clear(&mut self) {
        self.tt.clear();
    }

    /// Solve a position reusing this solver's TT. Same return shape as the
    /// free `solve_tinue` function.
    ///
    /// `prefilter_margin` enables the **sweep-only** road-distance skip: a
    /// position whose attacker is further than `ceil(max_plies/2) + margin`
    /// squares from any road is reported `no_tinue` without searching.
    /// Negative (the default for callers that omit it as `-1`) disables it.
    /// It is a heuristic and can in principle skip a real tinue — pass it
    /// only from a full-game sweep, never from an explicit single-position
    /// check.
    ///
    /// Entries stored under different `scope` values live in separate TT
    /// namespaces, so flipping the scope mid-sweep is safe.
    pub fn solve(
        &mut self,
        tps: &str,
        size: u8,
        max_plies: u32,
        max_nodes: f64,
        scope: Option<String>,
        prefilter_margin: f64,
    ) -> JsValue {
        let pos = match parse_position(tps, size) {
            Ok(p) => p,
            Err(message) => return error_response(message),
        };
        let scope = match parse_scope(scope) {
            Ok(s) => s,
            Err(message) => return error_response(message),
        };
        let limits = Limits {
            max_plies,
            max_nodes: parse_max_nodes(max_nodes),
            scope,
            prefilter_margin: parse_prefilter_margin(prefilter_margin),
            ..Default::default()
        };
        let (result, stats) = tinue::solve_with_tt(&pos, &limits, &mut self.tt);
        to_jsvalue(build_response(result, stats))
    }

    /// Run exactly one iteration at `depth` plies. Use repeatedly with
    /// increasing odd depths to drive iterative deepening from JS so
    /// per-depth progress can be surfaced to the UI. The TT survives
    /// across calls, so earlier-depth work warms the cache for later
    /// depths just as the internal iterative-deepening loop would.
    pub fn solve_at_depth(
        &mut self,
        tps: &str,
        size: u8,
        depth: u32,
        max_nodes: f64,
        find_all_winners: bool,
        scope: Option<String>,
    ) -> JsValue {
        let pos = match parse_position(tps, size) {
            Ok(p) => p,
            Err(message) => return error_response(message),
        };
        let scope = match parse_scope(scope) {
            Ok(s) => s,
            Err(message) => return error_response(message),
        };
        let limits = Limits {
            max_plies: depth,
            max_nodes: parse_max_nodes(max_nodes),
            find_all_winners,
            scope,
            ..Default::default()
        };
        let (result, stats) = tinue::solve_one_depth(&pos, depth, &mut self.tt, &limits);
        to_jsvalue(build_response(result, stats))
    }

    /// Score every legal move at `tps` against the warm TT from
    /// `attacker`'s perspective (`attacker_p1 = true` → P1 is attacker).
    /// Pure TT lookup — no fresh search. Run a `solve`/`solve_at_depth`
    /// first to populate the TT; call this on every UI navigation tick.
    /// Returns a `[{ move, kind, ... }]` array; see [`MoveScoreEntryKind`].
    ///
    /// Pass the same `scope` the populating solve used — verdicts are stored
    /// per scope, so a mismatch yields `unknown` for every move rather than
    /// another scope's answers.
    pub fn score_moves(
        &self,
        tps: &str,
        size: u8,
        attacker_p1: bool,
        scope: Option<String>,
    ) -> JsValue {
        let pos = match parse_position(tps, size) {
            Ok(p) => p,
            Err(message) => return error_response(message),
        };
        let scope = match parse_scope(scope) {
            Ok(s) => s,
            Err(message) => return error_response(message),
        };
        let attacker = if attacker_p1 { Player::P1 } else { Player::P2 };
        let scores = tinue::score_moves(&pos, attacker, scope, &self.tt);
        let entries: Vec<MoveScoreEntry> = scores
            .into_iter()
            .map(|s| MoveScoreEntry {
                mv: s.mv.to_string(),
                kind: match s.kind {
                    MoveScoreKind::Win { plies } => MoveScoreEntryKind::Win { plies },
                    MoveScoreKind::Loss { plies } => MoveScoreEntryKind::Loss { plies },
                    MoveScoreKind::NoWin { searched } => MoveScoreEntryKind::NoWin { searched },
                    MoveScoreKind::Flat { outcome } => MoveScoreEntryKind::Flat {
                        outcome: flat_outcome_str(outcome),
                    },
                    MoveScoreKind::Unknown => MoveScoreEntryKind::Unknown,
                },
            })
            .collect();
        to_js(&entries)
    }
}
