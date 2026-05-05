/*
 * syntaks, a TEI Tak engine — wasm bindings.
 */

use crate::board::Position;
use crate::core::SIZE;
use crate::tinue::{self, AbortReason, Limits, TinueResult};
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

/// Solve for a tinue at the position described by `tps` on a board of the
/// given `size`. `max_plies` caps iterative deepening (odd values are
/// natural; even values round down internally). `max_nodes` is a budget
/// guard — pass 0 for no cap. Returns a JS object with shape
/// `{ outcome: { kind, ... }, nodes }`.
#[wasm_bindgen]
pub fn solve_tinue(tps: &str, size: u8, max_plies: u32, max_nodes: f64) -> JsValue {
    let response = match run(tps, size, max_plies, max_nodes) {
        Ok(r) => r,
        Err(message) => SolveResponse {
            outcome: SolveOutcome::Error { message },
            nodes: 0,
        },
    };
    serde_wasm_bindgen::to_value(&response).unwrap_or(JsValue::NULL)
}

fn run(tps: &str, size: u8, max_plies: u32, max_nodes: f64) -> Result<SolveResponse, String> {
    if !(5..=7).contains(&size) {
        return Err(format!("unsupported size {} (only 5/6/7)", size));
    }
    SIZE.store(size, Ordering::Release);

    let parts: Vec<&str> = tps.split_whitespace().collect();
    let pos = Position::from_tps_parts(&parts).map_err(|e| format!("tps parse: {:?}", e))?;

    let max_nodes = if max_nodes <= 0.0 || !max_nodes.is_finite() {
        u64::MAX
    } else {
        max_nodes.min(u64::MAX as f64) as u64
    };

    let limits = Limits {
        max_plies,
        max_nodes,
        ..Default::default()
    };
    let (result, stats) = tinue::solve(&pos, &limits);

    let outcome = match result {
        TinueResult::Tinue { plies, pv } => SolveOutcome::Tinue {
            plies,
            pv: pv.iter().map(|m| m.to_string()).collect(),
        },
        TinueResult::NoTinue { searched_plies } => SolveOutcome::NoTinue { searched_plies },
        TinueResult::Aborted { reason, searched_plies } => SolveOutcome::Aborted {
            reason: match reason {
                AbortReason::Cancelled => "cancelled",
                AbortReason::Nodes => "nodes",
                AbortReason::Depth => "depth",
            },
            searched_plies,
        },
    };

    Ok(SolveResponse {
        outcome,
        nodes: stats.nodes,
    })
}
