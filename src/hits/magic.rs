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

use crate::bitboard::Bitboard;
use crate::core::{Direction, Square};
use crate::hits::common::{generate_mask, pdep};
use crate::hits::naive::find_hits_naive;

// Magic constants are 6x6-specific. Only the first 36 entries are used;
// the rest are sentinel zeros so the array fits Square::MAX_COUNT.
#[rustfmt::skip]
const MAGICS: [u64; Square::MAX_COUNT] = {
    let mut m = [0u64; Square::MAX_COUNT];
    let src: [u64; 36] = [
        0x0200204004001181, 0x0101002001108a00, 0x0208081002804803, 0x0402240404000200, 0x0809088808000200, 0x2400402a20000220,
        0x0308004090000108, 0x0040c210003200c2, 0x1610402001080248, 0x00610201018a8000, 0x80204a0800000414, 0x2008010200211100,
        0x45002001488c0208, 0x0181101002501800, 0x1610402001080248, 0x1203040202044100, 0x0420410041830181, 0x0201800440020001,
        0x004c018022401088, 0x008100c010400084, 0x1480410500001060, 0x0020160180100012, 0x0420410041830181, 0x808102010e000001,
        0x1200104200040800, 0x04040c1102000008, 0x0142008202060401, 0x0241010402000ab0, 0xc304081000400002, 0x2008010200211100,
        0xc4480004b0001002, 0x4404012a10004040, 0x1008181008018890, 0x000400881c000000, 0x0802894040220001, 0x8491034420002200,
    ];
    let mut i = 0;
    while i < 36 { m[i] = src[i]; i += 1; }
    m
};

#[rustfmt::skip]
const SHIFTS: [u32; Square::MAX_COUNT] = {
    let mut s = [64u32; Square::MAX_COUNT];
    let src: [u32; 36] = [
        56, 57, 57, 57, 57, 56,
        57, 58, 58, 58, 58, 57,
        57, 58, 58, 58, 58, 57,
        57, 58, 58, 58, 58, 57,
        57, 58, 58, 58, 58, 57,
        56, 57, 57, 57, 57, 56,
    ];
    let mut i = 0;
    while i < 36 { s[i] = src[i]; i += 1; }
    s
};

#[derive(Copy, Clone, Debug)]
struct SquareData {
    inv_mask: u64,
    offset: usize,
}

impl SquareData {
    const fn new() -> Self {
        Self { inv_mask: 0, offset: 0 }
    }
}

struct Data {
    squares: [SquareData; Square::MAX_COUNT],
    table_size: usize,
}

const MAGIC_SIZE: usize = 6;
const MAGIC_SQ_COUNT: usize = MAGIC_SIZE * MAGIC_SIZE;

const SQUARE_DATA: Data = {
    let mut squares = [SquareData::new(); Square::MAX_COUNT];
    let mut table_size = 0;

    let mut idx: u8 = 0;
    while (idx as usize) < MAGIC_SQ_COUNT {
        let sq = Square::from_raw(idx).unwrap();
        let square_data = &mut squares[sq.idx()];

        square_data.inv_mask = !generate_mask(sq, MAGIC_SIZE);

        square_data.offset = table_size;
        table_size += 1 << (64 - SHIFTS[sq.idx()]);

        idx += 1;
    }

    Data { squares, table_size }
};

// Native uses `#[static_init::dynamic]` so the table is built before main
// and lives in .data (zero per-access overhead). That crate has no wasm
// support — its build chooses an initializer-section linkage per platform
// and bails on `wasm32-unknown-unknown`. On wasm we lazy-init the same
// data into a Vec via `OnceLock`; first call pays the build cost (~few MB
// of writes), all subsequent calls are a load + branch.
//
// Both sides go through `hits_table()` below so callers don't care which
// variant they're using.

#[cfg(not(target_arch = "wasm32"))]
#[static_init::dynamic]
static HITS: [super::Hits; SQUARE_DATA.table_size] = build_hits_array();

#[cfg(target_arch = "wasm32")]
static HITS_CELL: std::sync::OnceLock<Vec<super::Hits>> = std::sync::OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn build_hits_array() -> [super::Hits; SQUARE_DATA.table_size] {
    let mut result = [[(0, Square::A1); Direction::COUNT]; SQUARE_DATA.table_size];
    let mut filled = [false; SQUARE_DATA.table_size];
    fill_hits(&mut result, &mut filled);
    result
}

#[cfg(target_arch = "wasm32")]
fn build_hits_vec() -> Vec<super::Hits> {
    let mut result = vec![[(0, Square::A1); Direction::COUNT]; SQUARE_DATA.table_size];
    let mut filled = vec![false; SQUARE_DATA.table_size];
    fill_hits(&mut result, &mut filled);
    result
}

#[inline]
fn fill_hits(result: &mut [super::Hits], filled: &mut [bool]) {
    for sq in Square::all() {
        let sq_data = &SQUARE_DATA.squares[sq.idx()];

        let magic = MAGICS[sq.idx()];
        let shift = SHIFTS[sq.idx()];

        let mask = !sq_data.inv_mask;

        let max_entries = 1 << mask.count_ones();
        for i in 0..max_entries {
            let blockers = Bitboard::from_raw(pdep(i as u64, mask));

            let idx = sq_data.offset + calc_idx(blockers, sq_data.inv_mask, magic, shift);

            if filled[idx] {
                continue;
            }

            result[idx] = find_hits_naive(blockers, sq);
            filled[idx] = true;
        }
    }
}

#[inline]
fn hits_table() -> &'static [super::Hits] {
    #[cfg(not(target_arch = "wasm32"))]
    {
        &*HITS
    }
    #[cfg(target_arch = "wasm32")]
    {
        HITS_CELL.get_or_init(build_hits_vec)
    }
}

/// Force the magic tables to be built now. Native does this before main
/// via `#[static_init::dynamic]`; on wasm the first hits query would
/// otherwise stall to build a multi-MB lookup table. Callers that know
/// they're about to start a search can call this to amortize the cost.
#[inline]
pub fn preload() {
    let _ = hits_table();
}

#[must_use]
fn calc_idx(blockers: Bitboard, inv_mask: u64, magic: u64, shift: u32) -> usize {
    ((blockers.raw() | inv_mask).wrapping_mul(magic) >> shift) as usize
}

#[must_use]
pub fn find_hit_for_dir_magic(blockers: Bitboard, start: Square, dir: Direction) -> super::Hit {
    find_hits_magic(blockers, start)[dir.idx()]
}

#[must_use]
pub(super) fn find_hits_magic(blockers: Bitboard, start: Square) -> super::Hits {
    let magic = MAGICS[start.idx()];
    let shift = SHIFTS[start.idx()];

    let sq_data = &SQUARE_DATA.squares[start.idx()];

    let idx = calc_idx(blockers, sq_data.inv_mask, magic, shift);
    hits_table()[sq_data.offset + idx]
}
