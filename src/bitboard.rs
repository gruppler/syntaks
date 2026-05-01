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

use crate::core::*;
use std::ops::*;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Bitboard {
    raw: u64,
}

impl Bitboard {
    /// Runtime bit-mask covering valid squares for the current board size.
    #[must_use]
    #[inline]
    pub fn current_mask() -> u64 {
        (1u64 << current_size_sq()) - 1
    }

    /// Compile-time mask for a specific board size. Useful for per-size
    /// const tables.
    #[must_use]
    pub const fn mask_for(n: usize) -> u64 {
        if n >= 64 { u64::MAX } else { (1u64 << (n * n)) - 1 }
    }

    #[must_use]
    pub fn upper_edge() -> Self {
        let n = current_size() as u32;
        // top rank: bits N*(N-1) .. N*N - 1
        let raw = ((1u64 << n) - 1) << (n * (n - 1));
        Self { raw }
    }

    #[must_use]
    pub fn lower_edge() -> Self {
        let n = current_size() as u32;
        Self { raw: (1u64 << n) - 1 }
    }

    #[must_use]
    pub fn left_edge() -> Self {
        let n = current_size() as u32;
        let mut raw = 0u64;
        let mut i = 0;
        while i < n {
            raw |= 1u64 << (i * n);
            i += 1;
        }
        Self { raw }
    }

    #[must_use]
    pub fn right_edge() -> Self {
        let n = current_size() as u32;
        let mut raw = 0u64;
        let mut i = 0;
        while i < n {
            raw |= 1u64 << (i * n + n - 1);
            i += 1;
        }
        Self { raw }
    }

    /// Const edge masks for a specific board size.
    #[must_use]
    pub const fn upper_edge_const(n: u32) -> Self {
        let raw = ((1u64 << n) - 1) << (n * (n - 1));
        Self { raw }
    }

    #[must_use]
    pub const fn lower_edge_const(n: u32) -> Self {
        Self { raw: (1u64 << n) - 1 }
    }

    #[must_use]
    pub const fn left_edge_const(n: u32) -> Self {
        let mut raw = 0u64;
        let mut i = 0;
        while i < n {
            raw |= 1u64 << (i * n);
            i += 1;
        }
        Self { raw }
    }

    #[must_use]
    pub const fn right_edge_const(n: u32) -> Self {
        let mut raw = 0u64;
        let mut i = 0;
        while i < n {
            raw |= 1u64 << (i * n + n - 1);
            i += 1;
        }
        Self { raw }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self { raw: 0 }
    }

    /// Construct a bitboard, masking off bits outside the current board size.
    #[must_use]
    pub fn from_raw(raw: u64) -> Self {
        Self { raw: raw & Self::current_mask() }
    }

    /// Const constructor that does not apply any size mask. Callers must
    /// ensure no bits outside the active board are set.
    #[must_use]
    pub const fn from_raw_unmasked(raw: u64) -> Self {
        Self { raw }
    }

    #[must_use]
    pub fn edge(dir: Direction) -> Self {
        match dir {
            Direction::Up => Self::upper_edge(),
            Direction::Down => Self::lower_edge(),
            Direction::Left => Self::left_edge(),
            Direction::Right => Self::right_edge(),
        }
    }

    #[must_use]
    pub const fn edge_const(dir: Direction, n: u32) -> Self {
        match dir {
            Direction::Up => Self::upper_edge_const(n),
            Direction::Down => Self::lower_edge_const(n),
            Direction::Left => Self::left_edge_const(n),
            Direction::Right => Self::right_edge_const(n),
        }
    }

    #[must_use]
    pub const fn raw(self) -> u64 {
        self.raw
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.raw == 0
    }

    #[must_use]
    pub const fn has_sq(self, sq: Square) -> bool {
        (self.raw & sq.bb_const().raw) != 0
    }

    #[must_use]
    pub const fn with_sq(self, sq: Square) -> Self {
        Self {
            raw: self.raw | sq.bb_const().raw,
        }
    }

    #[must_use]
    pub const fn without_sq(self, sq: Square) -> Self {
        Self {
            raw: self.raw & !sq.bb_const().raw,
        }
    }

    #[must_use]
    pub const fn with_sq_toggled(self, sq: Square) -> Self {
        Self {
            raw: self.raw ^ sq.bb_const().raw,
        }
    }

    pub const fn set_sq(&mut self, sq: Square) {
        self.raw |= sq.bb_const().raw;
    }

    pub const fn clear_sq(&mut self, sq: Square) {
        self.raw &= !sq.bb_const().raw;
    }

    pub const fn toggle_sq(&mut self, sq: Square) {
        self.raw ^= sq.bb_const().raw;
    }

    #[must_use]
    pub fn cmpl(self) -> Self {
        Self {
            raw: !self.raw & Self::current_mask(),
        }
    }

    /// Const variant of [`cmpl`] that takes board size explicitly.
    #[must_use]
    pub const fn cmpl_const(self, n: usize) -> Self {
        Self {
            raw: !self.raw & Self::mask_for(n),
        }
    }

    #[must_use]
    pub const fn and(self, other: Self) -> Self {
        Self {
            raw: self.raw & other.raw,
        }
    }

    #[must_use]
    pub const fn or(self, other: Self) -> Self {
        Self {
            raw: self.raw | other.raw,
        }
    }

    #[must_use]
    pub const fn xor(self, other: Self) -> Self {
        Self {
            raw: self.raw ^ other.raw,
        }
    }

    #[must_use]
    pub const fn shr(self, count: u32) -> Self {
        Self { raw: self.raw >> count }
    }

    #[must_use]
    pub fn shl(self, count: u32) -> Self {
        Self {
            raw: (self.raw << count) & Self::current_mask(),
        }
    }

    #[must_use]
    pub const fn shl_const(self, count: u32, n: usize) -> Self {
        Self {
            raw: (self.raw << count) & Self::mask_for(n),
        }
    }

    #[must_use]
    pub const fn lsb(self) -> Option<Square> {
        if self.is_empty() {
            None
        } else {
            Some(Square::from_raw(self.raw.trailing_zeros() as u8).unwrap())
        }
    }

    pub fn pop_lsb(&mut self) -> Option<Square> {
        let sq = self.lsb()?;
        self.raw &= self.raw - 1;
        Some(sq)
    }

    #[must_use]
    pub const fn popcount(self) -> u32 {
        self.raw.count_ones()
    }

    #[must_use]
    pub fn shift(self, dir: Direction) -> Self {
        match dir {
            Direction::Up => self.shl(dir.offset() as u32),
            Direction::Down => self.shr(-dir.offset() as u32),
            Direction::Left => self.and(Self::left_edge().cmpl()).shr(1),
            Direction::Right => self.and(Self::right_edge().cmpl()).shl(1),
        }
    }

    /// Const variant of [`shift`] that takes board size explicitly.
    #[must_use]
    pub const fn shift_const(self, dir: Direction, n: usize) -> Self {
        let n_u32 = n as u32;
        match dir {
            Direction::Up => self.shl_const(n_u32, n),
            Direction::Down => self.shr(n_u32),
            Direction::Left => self.and(Self::left_edge_const(n_u32).cmpl_const(n)).shr(1),
            Direction::Right => self.and(Self::right_edge_const(n_u32).cmpl_const(n)).shl_const(1, n),
        }
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        self.cmpl()
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.and(rhs)
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.or(rhs)
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        self.xor(rhs)
    }
}

impl Shr<u32> for Bitboard {
    type Output = Bitboard;

    fn shr(self, rhs: u32) -> Self::Output {
        self.shr(rhs)
    }
}

impl Shl<u32> for Bitboard {
    type Output = Bitboard;

    fn shl(self, rhs: u32) -> Self::Output {
        self.shl(rhs)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs
    }
}

impl BitXorAssign for Bitboard {
    fn bitxor_assign(&mut self, rhs: Self) {
        *self = *self ^ rhs
    }
}

impl ShrAssign<u32> for Bitboard {
    fn shr_assign(&mut self, rhs: u32) {
        *self = *self >> rhs;
    }
}

impl ShlAssign<u32> for Bitboard {
    fn shl_assign(&mut self, rhs: u32) {
        *self = *self << rhs;
    }
}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = Biterator;

    fn into_iter(self) -> Self::IntoIter {
        Biterator { board: self }
    }
}

pub struct Biterator {
    board: Bitboard,
}

impl Iterator for Biterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        self.board.pop_lsb()
    }
}
