//! Integration test exercising the tinue solver against a curated corpus
//! of real puzzles. The fixture (`tests/data/puzzles.txt`) carries TPS +
//! expected ply count, sampled from Tiltak's labelled puzzle database and
//! filtered to entries where the syntaks solver currently reproduces the
//! labelled depth. A regression here means the solver either missed a
//! known tinue or now claims one shorter than the canonical proof.
//!
//! Two tests:
//! * [`puzzles_fast`] — depths 3 and 5, always runs.
//! * [`puzzles_slow`] — adds depth 7, gated behind `#[ignore]`.
//!
//! Because the engine's board size lives in a global atomic (`core::SIZE`),
//! any two of these tests running concurrently in the same process would
//! race — one test's `SIZE.store` can land mid-solve in another and corrupt
//! board geometry. They therefore serialize on [`SIZE_LOCK`], mirroring the
//! unit tests in `src/tinue.rs`, so a default `cargo test` is safe without
//! `--test-threads=1`.

use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard};
use syntaks::board::Position;
use syntaks::core::SIZE;
use syntaks::tinue::{solve_with_tt, Limits, TinueResult, TinueScope, Tt};

/// Serializes tests that mutate the process-global board [`SIZE`].
/// Poison-tolerant so a failing assert in one test doesn't cascade into
/// spurious lock failures in the others.
static SIZE_LOCK: Mutex<()> = Mutex::new(());

#[must_use]
fn lock_size() -> MutexGuard<'static, ()> {
    SIZE_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

const FIXTURE: &str = include_str!("data/puzzles.txt");

struct PuzzleCase<'a> {
    size: u8,
    expected_plies: u32,
    tps: &'a str,
}

fn parse_fixture(max_plies: u32) -> Vec<PuzzleCase<'static>> {
    FIXTURE
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                return None;
            }
            let mut parts = line.splitn(3, ' ');
            let size: u8 = parts.next()?.parse().ok()?;
            let expected_plies: u32 = parts.next()?.parse().ok()?;
            let tps = parts.next()?;
            if expected_plies > max_plies {
                return None;
            }
            Some(PuzzleCase {
                size,
                expected_plies,
                tps,
            })
        })
        .collect()
}

fn run_corpus(max_plies: u32) {
    let _guard = lock_size();
    let cases = parse_fixture(max_plies);
    assert!(!cases.is_empty(), "fixture produced no cases");

    for case in cases {
        SIZE.store(case.size, Ordering::Release);
        let tps_parts: Vec<&str> = case.tps.split_whitespace().collect();
        let pos = Position::from_tps_parts(&tps_parts)
            .unwrap_or_else(|e| panic!("tps parse failed for {:?}: {:?}", case.tps, e));

        // Bound the search to the expected depth so a regression that
        // makes the solver miss a tinue surfaces as a timely failure
        // rather than an open-ended hang.
        let limits = Limits {
            max_plies: case.expected_plies,
            ..Default::default()
        };
        let mut tt = Tt::new(20);
        let (result, _stats) = solve_with_tt(&pos, &limits, &mut tt);

        match result {
            TinueResult::Tinue { plies, .. } => {
                assert!(
                    plies <= case.expected_plies,
                    "expected ≤ {} plies, got {} for {}",
                    case.expected_plies,
                    plies,
                    case.tps
                );
            }
            other => panic!(
                "expected Tinue (≤{} plies) for {}, got {:?}",
                case.expected_plies, case.tps, other
            ),
        }
    }
}

#[test]
fn puzzles_fast() {
    run_corpus(5);
}

/// `TinueScope::TakChain` searches a subset of the moves `Full` does, so its
/// results must be a subset too: every restricted tinue is a real tinue, and
/// it can never be *shorter* than the full search's shortest mate.
///
/// The converse is deliberately not asserted — the corpus may well contain
/// gap tinues, which full mode finds and restricted mode is defined to miss.
/// That asymmetry is the whole point of having two scopes.
#[test]
fn tak_chain_results_are_a_subset_of_full() {
    let _guard = lock_size();
    let cases = parse_fixture(5);
    assert!(!cases.is_empty(), "fixture produced no cases");

    let mut restricted_found = 0;

    for case in cases {
        SIZE.store(case.size, Ordering::Release);
        let tps_parts: Vec<&str> = case.tps.split_whitespace().collect();
        let pos = Position::from_tps_parts(&tps_parts)
            .unwrap_or_else(|e| panic!("tps parse failed for {:?}: {:?}", case.tps, e));

        let solve = |scope| {
            let limits = Limits {
                max_plies: case.expected_plies,
                scope,
                ..Default::default()
            };
            let mut tt = Tt::new(20);
            solve_with_tt(&pos, &limits, &mut tt).0
        };

        let chain_plies = match solve(TinueScope::TakChain) {
            TinueResult::Tinue { plies, .. } => plies,
            _ => continue,
        };
        restricted_found += 1;

        match solve(TinueScope::Full) {
            TinueResult::Tinue { plies, .. } => assert!(
                plies <= chain_plies,
                "full mode found {} plies but tak-chain found {} for {} — \
                 restricting the move set cannot shorten a mate",
                plies,
                chain_plies,
                case.tps
            ),
            other => panic!(
                "tak-chain proved a tinue in {} for {} but full mode returned {:?} — \
                 a restricted proof is always a real proof",
                chain_plies, case.tps, other
            ),
        }
    }

    assert!(
        restricted_found > 0,
        "no corpus position was a tak-chain tinue; the subset check proved nothing"
    );
}

#[test]
#[ignore]
fn puzzles_slow() {
    run_corpus(7);
}
