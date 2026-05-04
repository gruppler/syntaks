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
 */

pub mod bitboard;
pub mod board;
pub mod core;
pub mod hits;
pub mod keys;
pub mod movegen;
pub mod road;
pub mod takmove;
pub mod tinue;
pub mod util;

// Native-only modules: TEI loop, search, eval, perft, threading, etc.
#[cfg(not(target_arch = "wasm32"))]
pub mod correction;
#[cfg(not(target_arch = "wasm32"))]
pub mod eval;
#[cfg(not(target_arch = "wasm32"))]
pub mod history;
#[cfg(not(target_arch = "wasm32"))]
pub mod limit;
#[cfg(not(target_arch = "wasm32"))]
pub mod movepick;
#[cfg(not(target_arch = "wasm32"))]
pub mod node_counter;
#[cfg(not(target_arch = "wasm32"))]
pub mod perft;
#[cfg(not(target_arch = "wasm32"))]
pub mod search;
#[cfg(not(target_arch = "wasm32"))]
pub mod tei;
#[cfg(not(target_arch = "wasm32"))]
pub mod thread;
#[cfg(not(target_arch = "wasm32"))]
pub mod ttable;

#[cfg(target_arch = "wasm32")]
pub mod wasm;
