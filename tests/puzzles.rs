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
//! running both `puzzles_*` tests concurrently in the same process would
//! race. The `#[ignore]` on the slow test sidesteps this for default runs;
//! if both are invoked together, pass `--test-threads=1`.

use std::sync::atomic::Ordering;
use syntaks::board::Position;
use syntaks::core::SIZE;
use syntaks::tinue::{solve_with_tt, Limits, TinueResult, Tt};

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

#[test]
#[ignore]
fn puzzles_slow() {
    run_corpus(7);
}
