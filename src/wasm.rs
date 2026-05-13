/*
 * syntaks, a TEI Tak engine — wasm bindings.
 */

use crate::board::Position;
use crate::core::SIZE;
use crate::tinue::{self, AbortReason, Limits, TinueResult, Tt};
use serde::Serialize;
use std::sync::atomic::Ordering;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
#[serde(tag = "kind")]
enum SolveOutcome {
    #[serde(rename = "tinue")]
    Tinue { plies: u32, pv: Vec<String> },
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
        TinueResult::Tinue { plies, pv } => SolveOutcome::Tinue {
            plies,
            pv: pv.iter().map(|m| m.to_string()).collect(),
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

fn parse_position(tps: &str, size: u8) -> Result<Position, String> {
    if !(5..=7).contains(&size) {
        return Err(format!("unsupported size {} (only 5/6/7)", size));
    }
    SIZE.store(size, Ordering::Release);
    let parts: Vec<&str> = tps.split_whitespace().collect();
    Position::from_tps_parts(&parts).map_err(|e| format!("tps parse: {:?}", e))
}

fn to_jsvalue(response: SolveResponse) -> JsValue {
    serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
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
#[wasm_bindgen]
pub fn solve_tinue(tps: &str, size: u8, max_plies: u32, max_nodes: f64) -> JsValue {
    let pos = match parse_position(tps, size) {
        Ok(p) => p,
        Err(message) => return error_response(message),
    };
    let limits = Limits {
        max_plies,
        max_nodes: parse_max_nodes(max_nodes),
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
    pub fn solve(&mut self, tps: &str, size: u8, max_plies: u32, max_nodes: f64) -> JsValue {
        let pos = match parse_position(tps, size) {
            Ok(p) => p,
            Err(message) => return error_response(message),
        };
        let limits = Limits {
            max_plies,
            max_nodes: parse_max_nodes(max_nodes),
            ..Default::default()
        };
        let (result, stats) = tinue::solve_with_tt(&pos, &limits, &mut self.tt);
        to_jsvalue(build_response(result, stats))
    }
}
