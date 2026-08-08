//! Native tinue-solver CLI. Reads a TPS (size auto-detected from the rank
//! count) and prints a single JSON line to stdout with the verdict, ply count,
//! node count, and the internal solve wall-time in milliseconds.
//!
//! The output shape deliberately matches the `topaz-tinue` binary in
//! `../topaz-tinue-web` so the two engines can be diffed line-for-line — that
//! oracle is the correctness bar for `--scope tak-chain`.
//!
//! This is the test and benchmark harness for the solver, so it exposes every
//! mode the wasm bindings can reach. Anything reachable from one entry point
//! must be reachable from the other; otherwise native runs stop being a
//! faithful proxy for wasm behaviour.
//!
//! ```text
//! tinue "<tps>" [options]
//!
//!   --scope full|tak-chain    move set to search (default: full)
//!   --max-plies N             cap iterative deepening (default: solver's own)
//!   --max-nodes N             per-depth node budget; 0 = unlimited (default)
//!   --tt-bits N               TT size = 1<<N entries x 16 B (default: 20)
//!   --root-move PTN           restrict the attacker's first move to this one
//!   --prefilter-margin N      enable the sweep road-distance skip
//!   --batch                   read TPS lines from stdin, one JSON line out each
//!   --quiet                   suppress the per-depth stderr progress log
//! ```
//!
//! The TPS must be a single argument — quote it, it contains spaces.
//!
//! `--batch` exists so bulk differential runs do not pay process-spawn cost
//! per position; it takes no TPS argument and implies `--quiet`. Each input
//! line is one TPS, optionally prefixed by its size (the `genpos` output
//! format), and every line yields exactly one JSON result line so inputs and
//! outputs stay index-aligned even when a position fails to parse.
//!
//! `--scope tak-chain` restricts the search to strict tak chains. A `no_tinue`
//! under that scope means "no tak-chain tinue"; quiet tinues are excluded by
//! construction, so it is not evidence that the position is quiet.
//!
//! `--root-move` verifies one candidate without deep-searching every other
//! root move first. It is the fast way to confirm a suspected quiet tinue,
//! since move ordering would otherwise try that move last.
//!
//! Per-depth progress goes to stderr so a long search stays observable
//! without polluting the stdout JSON.

use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::Ordering;
use std::time::Instant;

use syntaks::board::{set_standard_reserves, Position};
use syntaks::core::SIZE;
use syntaks::takmove::Move;
use syntaks::tinue::{self, Limits, Stats, TinueResult, TinueScope, Tt};

const USAGE: &str = "usage: tinue \"<tps>\" [--scope full|tak-chain] [--max-plies N] \
                     [--max-nodes N] [--tt-bits N] [--root-move PTN] \
                     [--prefilter-margin N] [--batch] [--quiet]";

struct Args {
    tps: String,
    scope: TinueScope,
    max_plies: Option<u32>,
    max_nodes: Option<u64>,
    tt_bits: u32,
    root_move: Option<String>,
    prefilter_margin: Option<u32>,
    quiet: bool,
    batch: bool,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args {
        tps: String::new(),
        scope: TinueScope::Full,
        max_plies: None,
        max_nodes: None,
        tt_bits: 20,
        root_move: None,
        prefilter_margin: None,
        quiet: false,
        batch: false,
    };

    let mut positional: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        // Options take their value as the next argv entry; `next` centralises
        // the "flag given without a value" error so each arm stays one line.
        let mut next = |what: &str| -> Result<String, String> {
            i += 1;
            argv.get(i)
                .cloned()
                .ok_or_else(|| format!("{what} requires a value"))
        };
        match arg {
            "--scope" => {
                let v = next("--scope")?;
                args.scope = v
                    .parse()
                    .map_err(|_| format!("unknown scope {v:?} (expected full or tak-chain)"))?;
            }
            "--max-plies" => {
                let v = next("--max-plies")?;
                args.max_plies = Some(v.parse().map_err(|_| format!("bad --max-plies {v:?}"))?);
            }
            "--max-nodes" => {
                let v = next("--max-nodes")?;
                let n: u64 = v.parse().map_err(|_| format!("bad --max-nodes {v:?}"))?;
                // 0 spells "unlimited", matching the wasm binding's treatment
                // of a non-positive budget.
                args.max_nodes = (n > 0).then_some(n);
            }
            "--tt-bits" => {
                let v = next("--tt-bits")?;
                args.tt_bits = v.parse().map_err(|_| format!("bad --tt-bits {v:?}"))?;
            }
            "--root-move" => args.root_move = Some(next("--root-move")?),
            "--prefilter-margin" => {
                let v = next("--prefilter-margin")?;
                args.prefilter_margin =
                    Some(v.parse().map_err(|_| format!("bad --prefilter-margin {v:?}"))?);
            }
            "--quiet" => args.quiet = true,
            "--batch" => {
                args.batch = true;
                args.quiet = true;
            }
            "-h" | "--help" => return Err(USAGE.to_string()),
            other if other.starts_with('-') => return Err(format!("unknown option {other:?}")),
            other => {
                if positional.replace(other.to_string()).is_some() {
                    return Err("expected exactly one TPS argument (quote it)".to_string());
                }
            }
        }
        i += 1;
    }

    if args.batch {
        if positional.is_some() {
            return Err("--batch reads TPS from stdin; do not also pass one".to_string());
        }
    } else {
        args.tps = positional.ok_or_else(|| USAGE.to_string())?;
    }
    if !(10..=28).contains(&args.tt_bits) {
        return Err(format!("--tt-bits {} out of range (10..=28)", args.tt_bits));
    }
    Ok(args)
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            return ExitCode::from(2);
        }
    };

    if args.batch {
        return run_batch(&args);
    }

    match solve_one(args.tps.trim(), &args) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(msg) => {
            eprintln!("{msg}");
            ExitCode::FAILURE
        }
    }
}

/// Read one TPS per line from stdin and emit one JSON line each. Lines may
/// carry a leading size field (the `genpos` output format), which is ignored
/// — size is always inferred from the rank count, exactly as in single-shot
/// mode.
///
/// A failing line still produces an output line, as an `{"error": ...}`
/// object, so callers can zip inputs to outputs positionally without having
/// to track which ones dropped out.
fn run_batch(args: &Args) -> ExitCode {
    use std::io::BufRead;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut failures = 0u32;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("stdin read error: {e}");
                return ExitCode::FAILURE;
            }
        };
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Strip an optional leading size field: a bare integer followed by
        // the TPS. The TPS itself never begins with a lone integer token.
        let tps = match line.split_once(' ') {
            Some((head, rest)) if head.parse::<u8>().is_ok() => rest,
            _ => line,
        };

        match solve_one(tps.trim(), args) {
            Ok(json) => {
                let _ = writeln!(out, "{json}");
            }
            Err(msg) => {
                failures += 1;
                let escaped = msg.replace('\\', "\\\\").replace('"', "\\\"");
                let _ = writeln!(out, "{{\"engine\":\"syntaks\",\"error\":\"{escaped}\"}}");
            }
        }
        let _ = out.flush();
    }

    if failures > 0 {
        eprintln!("[tinue] {failures} line(s) failed");
    }
    ExitCode::SUCCESS
}

/// Solve a single TPS and render the result as one JSON line.
fn solve_one(tps: &str, args: &Args) -> Result<String, String> {
    let size = (tps.matches('/').count() + 1) as u8;
    if !(5..=7).contains(&size) {
        return Err(format!("unsupported size {size} (only 5/6/7)"));
    }

    // Global board geometry must be set before parsing or solving. Reserves
    // are a separate global from SIZE and do NOT follow it, so they have to
    // be installed explicitly or a 5x5 position is solved with the 6x6
    // reserve of 30 flats.
    SIZE.store(size, Ordering::Release);
    set_standard_reserves(size);

    let parts: Vec<&str> = tps.split_whitespace().collect();
    let pos = Position::from_tps_parts(&parts).map_err(|e| format!("tps parse error: {e:?}"))?;

    let mut limits = Limits {
        scope: args.scope,
        ..Default::default()
    };
    if let Some(mp) = args.max_plies {
        limits.max_plies = mp;
    }
    if let Some(mn) = args.max_nodes {
        limits.max_nodes = mn;
    }

    // Parsed per position because move syntax depends on the board size that
    // was just installed above.
    let root_move = match args.root_move.as_deref() {
        Some(s) => Some(
            s.parse::<Move>()
                .map_err(|e| format!("root_move parse error for {s:?}: {e:?}"))?,
        ),
        None => None,
    };

    let t0 = Instant::now();
    let (result, total_nodes) = solve_verbose(&pos, &limits, args, size, root_move);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (verdict, plies) = match &result {
        TinueResult::Tinue { plies, .. } => ("tinue", *plies as i64),
        TinueResult::NoTinue { searched_plies } => ("no_tinue", -(*searched_plies as i64)),
        TinueResult::Aborted { searched_plies, .. } => ("aborted", -(*searched_plies as i64)),
    };
    let pv = match &result {
        TinueResult::Tinue { pv, .. } => pv
            .iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(","),
        _ => String::new(),
    };

    // `road_distance` is reported as a diagnostic — it drives the sweep
    // pre-filter, and having it on every line makes it possible to select
    // contested positions when building differential-test sets.
    let road_distance = match tinue::road_distance(&pos, pos.stm()) {
        Some(d) => d.to_string(),
        None => "null".to_string(),
    };

    // Single machine-readable line. `plies` is negative when it reflects a
    // searched depth rather than a proven mate distance — that keeps the
    // field shape identical to the topaz oracle's output.
    Ok(format!(
        "{{\"engine\":\"syntaks\",\"size\":{size},\"scope\":\"{}\",\"verdict\":\"{verdict}\",\
         \"plies\":{plies},\"nodes\":{total_nodes},\"ms\":{ms:.3},\
         \"road_distance\":{road_distance},\"pv\":[{pv}]}}",
        args.scope.as_str()
    ))
}

/// Drive iterative deepening from the outside so each completed odd-ply depth
/// can be logged to stderr as it finishes. Uses a caller-sized TT (shared
/// across depths, exactly like the internal loop) so a large table can be
/// supplied for hard positions. Returns the final result and cumulative nodes.
fn solve_verbose(
    pos: &Position,
    limits: &Limits,
    args: &Args,
    size: u8,
    root_move: Option<Move>,
) -> (TinueResult, u64) {
    // The pre-filter belongs to the sweep, but the CLI exposes it so its
    // behaviour can be measured and regression-tested natively.
    if let Some(margin) = args.prefilter_margin
        && tinue::prefilter_skips(pos, limits.max_plies, margin)
    {
        if !args.quiet {
            eprintln!(
                "[tinue] pre-filter skip: road_distance {:?} exceeds horizon + margin {margin}",
                tinue::road_distance(pos, pos.stm())
            );
        }
        return (
            TinueResult::NoTinue {
                searched_plies: limits.max_plies,
            },
            0,
        );
    }

    let mut tt = Tt::new(args.tt_bits);
    let start = Instant::now();
    let mut total_nodes: u64 = 0;

    if !args.quiet {
        eprintln!(
            "[tinue] size {size}  scope {}  max_plies {}  tt_bits {} ({} MB){}",
            limits.scope.as_str(),
            limits.max_plies,
            args.tt_bits,
            ((1u64 << args.tt_bits) * 16) / (1 << 20),
            root_move.map_or(String::new(), |m| format!("  root_move {m}"))
        );
    }

    let mut depth = 1u32;
    while depth <= limits.max_plies {
        let d0 = Instant::now();
        let (result, stats): (TinueResult, Stats) = match root_move {
            Some(m) => tinue::solve_one_depth_root_move(pos, depth, &mut tt, limits, m),
            None => tinue::solve_one_depth(pos, depth, &mut tt, limits),
        };
        total_nodes += stats.nodes;
        if !args.quiet {
            eprintln!(
                "[tinue] depth {depth:>2}  +{:>13} nodes  cum {:>15}  {:>8.2}s (total {:>8.1}s)",
                stats.nodes,
                total_nodes,
                d0.elapsed().as_secs_f64(),
                start.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
        }

        match result {
            TinueResult::Tinue { .. } | TinueResult::Aborted { .. } => return (result, total_nodes),
            TinueResult::NoTinue { .. } => {}
        }
        depth += 2;
    }

    (
        TinueResult::NoTinue {
            searched_plies: limits.max_plies,
        },
        total_nodes,
    )
}
