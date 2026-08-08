//! Differential test for [`Position::spread_completes_road`].
//!
//! The fast path exists so the tinue solver can ask "does this spread finish
//! my road?" without building a whole successor position. Its correctness bar
//! is exact agreement with the thing it replaces:
//!
//! ```text
//! pos.spread_completes_road(mv) == pos.apply_move(mv).has_road(pos.stm())
//! ```
//!
//! Coverage comes from the puzzle corpus plus every position one and two
//! plies deep from it. That reaches the cases the shortcut has to reason
//! about by hand — mixed-ownership stacks, walls and capstones as the carried
//! top piece, a capstone flattening a wall on the final square, and spreads
//! that empty their source square.
//!
//! Serializes on [`SIZE_LOCK`] for the same reason `tests/puzzles.rs` does:
//! board geometry lives in a process-global atomic.

use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};
use syntaks::board::{set_standard_reserves, Position};
use syntaks::core::{Player, SIZE};
use syntaks::movegen::generate_moves;

static SIZE_LOCK: Mutex<()> = Mutex::new(());

#[must_use]
fn lock_size() -> MutexGuard<'static, ()> {
    SIZE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const FIXTURE: &str = include_str!("data/puzzles.txt");

/// Assert the shortcut agrees with `apply_move` on every spread available in
/// `pos`. Returns how many spreads were checked, so the caller can prove the
/// walk actually exercised something.
fn check_spreads(pos: &Position) -> usize {
    let stm = pos.stm();
    let mut buf = Vec::new();
    generate_moves(&mut buf, pos);

    let mut checked = 0;
    for &mv in &buf {
        let after = pos.apply_move(mv);

        // `roads_after` is checked for BOTH players, not just the mover.
        // Move ordering needs the opponent's road bitboard too — the
        // defender ordering scores a reply by how many attacker road
        // pieces it strips — and the two sides are not symmetric: a drop
        // that adds a square to one player removes it from the other.
        for player in [Player::P1, Player::P2] {
            let expected = after.roads(player);
            let actual = pos.roads_after(mv, player);
            assert_eq!(
                actual,
                expected,
                "roads_after disagreed with apply_move\n  \
                 move: {mv:?}\n  player: {player:?}\n  tps: {}\n  \
                 expected (oracle): {expected:?}\n  actual: {actual:?}",
                pos.tps(),
            );
        }

        if !mv.is_spread() {
            continue;
        }
        checked += 1;

        let expected = after.has_road(stm);
        let actual = pos.spread_completes_road(mv);

        assert_eq!(
            actual,
            expected,
            "spread_completes_road disagreed with apply_move\n  \
             move: {mv:?}\n  tps: {}\n  expected (oracle): {expected}\n  actual: {actual}",
            pos.tps(),
        );
    }
    checked
}

/// Plies to play on past each corpus position. Playing on matters more than
/// the corpus itself: it builds taller, more mixed stacks than the puzzles
/// start with, which is where hand-derived drop logic is most likely to be
/// wrong.
///
/// Debug builds re-check every `apply_move` invariant on top of each
/// comparison and run unoptimised, which is orders of magnitude slower — two
/// plies there does not finish in any useful time. One ply still reaches
/// covered walls and freshly-buried flats; release keeps the full sweep and
/// still runs in seconds.
const DEPTH: u32 = if cfg!(debug_assertions) { 1 } else { 2 };

/// Minimum spreads the walk must reach before the result means anything.
const MIN_SPREADS: usize = if cfg!(debug_assertions) { 10_000 } else { 1_000_000 };

fn walk(pos: &Position, depth: u32, positions: &mut usize, spreads: &mut usize) {
    *positions += 1;
    *spreads += check_spreads(pos);

    if depth == 0 {
        return;
    }

    let mut moves = Vec::new();
    generate_moves(&mut moves, pos);
    for &mv in &moves {
        walk(&pos.apply_move(mv), depth - 1, positions, spreads);
    }
}

#[test]
fn spread_completes_road_matches_apply_move() {
    let _guard = lock_size();

    let mut positions_walked = 0usize;
    let mut spreads_checked = 0usize;

    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let Some(size) = parts.next().and_then(|s| s.parse::<u8>().ok()) else {
            continue;
        };
        let Some(_plies) = parts.next() else { continue };
        let Some(tps) = parts.next() else { continue };

        SIZE.store(size, Ordering::Release);
        set_standard_reserves(size);

        let tps_parts: Vec<&str> = tps.split_whitespace().collect();
        let Ok(pos) = Position::from_tps_parts(&tps_parts) else {
            continue;
        };

        walk(&pos, DEPTH, &mut positions_walked, &mut spreads_checked);
    }

    assert!(
        positions_walked > 0,
        "fixture produced no positions; the differential proved nothing"
    );
    assert!(
        spreads_checked > MIN_SPREADS,
        "only {spreads_checked} spreads checked across {positions_walked} positions — \
         too few to trust the fast path"
    );

    eprintln!("checked {spreads_checked} spreads across {positions_walked} positions");
}
