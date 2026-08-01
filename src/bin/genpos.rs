//! Random legal-position generator, for differential testing of the tinue
//! solver against the `topaz-tinue` oracle.
//!
//! Plays a fixed number of uniformly-random legal moves from the start
//! position and prints `<size> <tps>` per line. Positions where the game has
//! already ended (a road exists, or flat count resolved) are skipped — they
//! are not tinue candidates and both engines short-circuit them.
//!
//! ```text
//! genpos [--size 5|6|7] [--plies N] [--count N] [--seed N]
//! ```
//!
//! The seed makes runs reproducible, so a differential failure can be
//! replayed exactly. The RNG is splitmix64 written out here rather than
//! pulled in as a dependency: this is a test tool, and its only requirement
//! is that the same seed gives the same positions.

use std::process::ExitCode;
use std::sync::atomic::Ordering;

use syntaks::board::{FlatCountOutcome, Position};
use syntaks::core::{Player, SIZE};
use syntaks::movegen::generate_moves;
use syntaks::takmove::Move;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut size: u8 = 5;
    let mut plies: u32 = 20;
    let mut count: u32 = 100;
    let mut seed: u64 = 1;

    let mut i = 0;
    while i < argv.len() {
        let value = || -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} requires a value", argv[i]))
        };
        let parsed = match argv[i].as_str() {
            "--size" => value().and_then(|v| {
                v.parse().map(|x| size = x).map_err(|_| format!("bad size {v:?}"))
            }),
            "--plies" => value().and_then(|v| {
                v.parse().map(|x| plies = x).map_err(|_| format!("bad plies {v:?}"))
            }),
            "--count" => value().and_then(|v| {
                v.parse().map(|x| count = x).map_err(|_| format!("bad count {v:?}"))
            }),
            "--seed" => value().and_then(|v| {
                v.parse().map(|x| seed = x).map_err(|_| format!("bad seed {v:?}"))
            }),
            other => Err(format!(
                "unknown option {other:?}; usage: genpos [--size N] [--plies N] [--count N] [--seed N]"
            )),
        };
        if let Err(msg) = parsed {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
        i += 2;
    }

    if !(5..=7).contains(&size) {
        eprintln!("unsupported size {size} (only 5/6/7)");
        return ExitCode::from(2);
    }
    SIZE.store(size, Ordering::Release);

    let mut rng = SplitMix64(seed);
    let mut moves: Vec<Move> = Vec::with_capacity(256);
    let mut emitted = 0u32;

    // Bound the attempt count so a pathological parameter set (e.g. a ply
    // depth at which nearly every game has already ended) terminates instead
    // of spinning.
    let mut attempts = 0u32;
    while emitted < count && attempts < count * 20 {
        attempts += 1;
        let mut pos = Position::startpos();
        let mut alive = true;

        for _ in 0..plies {
            if pos.has_road(Player::P1)
                || pos.has_road(Player::P2)
                || !matches!(pos.count_flats(), FlatCountOutcome::None)
            {
                alive = false;
                break;
            }
            generate_moves(&mut moves, &pos);
            if moves.is_empty() {
                alive = false;
                break;
            }
            let pick = moves[rng.below(moves.len())];
            pos = pos.apply_move(pick);
        }

        // Re-check after the final move: the loop's guard runs before each
        // move, so the position reached by the last one is still untested.
        if !alive
            || pos.has_road(Player::P1)
            || pos.has_road(Player::P2)
            || !matches!(pos.count_flats(), FlatCountOutcome::None)
        {
            continue;
        }

        println!("{size} {}", pos.tps());
        emitted += 1;
    }

    if emitted < count {
        eprintln!("warning: emitted {emitted}/{count} positions before hitting the attempt cap");
    }
    ExitCode::SUCCESS
}
