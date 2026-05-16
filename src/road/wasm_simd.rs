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
use core::arch::wasm32::*;

#[must_use]
#[target_feature(enable = "simd128")]
pub(super) fn has_road(road_occ: u64, up: u64, down: u64, left: u64, right: u64) -> bool {
    // Pack 4 frontiers across two v128 registers (2× u64 lanes each), mirroring
    // the SSE 6x6 implementation.
    let mut masks_ul = u64x2(left, up);
    let mut masks_dr = u64x2(right, down);

    let left_edge = u64x2_splat(Bitboard::left_edge_const(6).raw());
    let right_edge = u64x2_splat(Bitboard::right_edge_const(6).raw());
    let road_occ = u64x2_splat(road_occ);

    let calc_next = |masks: v128| -> v128 {
        let up_shift = u64x2_shl(masks, 6);
        let down_shift = u64x2_shr(masks, 6);
        let ud = v128_or(up_shift, down_shift);

        // _mm_andnot_si128(edge, x << 1) → (x << 1) & !edge → v128_andnot((x<<1), edge)
        let left_shift = v128_andnot(u64x2_shl(masks, 1), left_edge);
        let right_shift = v128_andnot(u64x2_shr(masks, 1), right_edge);
        let lr = v128_or(left_shift, right_shift);

        v128_and(v128_or(ud, lr), road_occ)
    };

    masks_ul = calc_next(masks_ul);
    masks_dr = calc_next(masks_dr);

    loop {
        let next_ul = calc_next(masks_ul);
        let next_dr = calc_next(masks_dr);

        // SSE: _mm_testz_si128(next_ul, next_dr) == 0  →  (next_ul & next_dr) != 0
        // up-frontier meets down-frontier, or left meets right (lane-aligned).
        if v128_any_true(v128_and(next_ul, next_dr)) {
            return true;
        }

        // Mirror the SSE liveness check: an axis is dead once either of its two
        // frontiers stops growing (the stuck side can never meet the other),
        // and if both axes are dead, no road is possible. The AND across the two
        // registers folds "both lanes per axis grew" into a non-zero mask only
        // when at least one axis still has both frontiers expanding.
        let new_ul = i64x2_gt(next_ul, masks_ul);
        let new_dr = i64x2_gt(next_dr, masks_dr);

        if !v128_any_true(v128_and(new_ul, new_dr)) {
            return false;
        }

        masks_ul = next_ul;
        masks_dr = next_dr;
    }
}
