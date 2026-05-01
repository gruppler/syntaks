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

#[cfg(target_feature = "avx2")]
mod avx2;

#[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.2"))]
mod sse;

use crate::bitboard::Bitboard;

#[must_use]
pub fn has_road(road_occ: Bitboard) -> bool {
    let n = crate::core::current_size() as u32;
    let upper_edge = Bitboard::upper_edge().raw();
    let lower_edge = Bitboard::lower_edge().raw();
    let left_edge = Bitboard::left_edge().raw();
    let right_edge = Bitboard::right_edge().raw();

    let road_occ = road_occ.raw();

    let up = road_occ & upper_edge;
    let down = road_occ & lower_edge;
    let left = road_occ & left_edge;
    let right = road_occ & right_edge;

    let up = up | (up >> n & road_occ);
    let down = down | (down << n & road_occ);
    let left = left | (left << 1 & road_occ);
    let right = right | (right >> 1 & road_occ);

    #[cfg(target_feature = "avx2")]
    if n == 6 {
        // SAFETY: AVX2 path is hardcoded for 6x6 stride.
        return unsafe { avx2::has_road(road_occ, up, down, left, right) };
    }

    #[cfg(all(not(target_feature = "avx2"), target_feature = "sse4.2"))]
    if n == 6 {
        return unsafe { sse::has_road(road_occ, up, down, left, right) };
    }

    has_road_scalar(n, road_occ, up, down, left, right, left_edge, right_edge)
}

#[must_use]
fn has_road_scalar(
    n: u32,
    road_occ: u64,
    mut up: u64,
    mut down: u64,
    mut left: u64,
    mut right: u64,
    left_edge: u64,
    right_edge: u64,
) -> bool {
    loop {
        let next_up = (up << n | up >> n | (up & !left_edge) >> 1 | (up & !right_edge) << 1) & road_occ;
        let next_down = (down << n | down >> n | (down & !left_edge) >> 1 | (down & !right_edge) << 1) & road_occ;
        let next_left = (left << n | left >> n | (left & !left_edge) >> 1 | (left & !right_edge) << 1) & road_occ;
        let next_right = (right << n | right >> n | (right & !left_edge) >> 1 | (right & !right_edge) << 1) & road_occ;

        if (next_up & next_down) != 0 || (next_left & next_right) != 0 {
            return true;
        }

        let progressed =
            (next_up & !up) | (next_down & !down) | (next_left & !left) | (next_right & !right);
        if progressed == 0 {
            return false;
        }

        up = next_up;
        down = next_down;
        left = next_left;
        right = next_right;
    }
}
