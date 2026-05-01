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
use std::fmt::{Display, Formatter, Write};
use std::str::FromStr;
use std::sync::atomic::{AtomicU8, Ordering};

pub const MIN_SIZE: u8 = 5;
pub const MAX_SIZE: u8 = 7;
pub const DEFAULT_SIZE: u8 = 6;
pub const MAX_SQ: usize = (MAX_SIZE as usize) * (MAX_SIZE as usize);

pub static SIZE: AtomicU8 = AtomicU8::new(DEFAULT_SIZE);

#[must_use]
#[inline]
pub fn current_size() -> u8 {
    SIZE.load(Ordering::Relaxed)
}

#[must_use]
#[inline]
pub fn current_size_sq() -> usize {
    let n = current_size() as usize;
    n * n
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    pub const COUNT: usize = 2;

    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::P1),
            1 => Some(Self::P2),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn sign(self) -> i32 {
        match self {
            Self::P1 => 1,
            Self::P2 => -1,
        }
    }

    #[must_use]
    pub const fn flip(self) -> Self {
        Self::from_raw(self as u8 ^ 0x1).unwrap()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum PieceType {
    Flat,
    Wall,
    Capstone,
}

impl PieceType {
    pub const COUNT: usize = 3;

    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Flat),
            1 => Some(Self::Wall),
            2 => Some(Self::Capstone),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn with_player(self, player: Player) -> Piece {
        Piece::from_raw((self.raw() << 1) | player.raw()).unwrap()
    }

    #[must_use]
    pub const fn is_blocker(self) -> bool {
        self.raw() > 0
    }

    #[must_use]
    pub const fn is_road(self) -> bool {
        self.raw() & 0b1 == 0
    }
}

impl Display for PieceType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PieceType::Flat => f.write_char('F'),
            PieceType::Wall => f.write_char('S'),
            PieceType::Capstone => f.write_char('C'),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Piece {
    P1Flat,
    P2Flat,
    P1Wall,
    P2Wall,
    P1Capstone,
    P2Capstone,
}

impl Piece {
    pub const COUNT: usize = 6;

    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::P1Flat),
            1 => Some(Self::P2Flat),
            2 => Some(Self::P1Wall),
            3 => Some(Self::P2Wall),
            4 => Some(Self::P1Capstone),
            5 => Some(Self::P2Capstone),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[must_use]
    pub const fn player(self) -> Player {
        Player::from_raw(self.raw() & 0x1).unwrap()
    }

    #[must_use]
    pub const fn piece_type(self) -> PieceType {
        PieceType::from_raw(self.raw() >> 1).unwrap()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub const COUNT: usize = 4;

    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Up),
            1 => Some(Self::Down),
            2 => Some(Self::Left),
            3 => Some(Self::Right),
            _ => None,
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self as u8
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[must_use]
    pub fn offset(self) -> i8 {
        let n = current_size() as i8;
        match self {
            Direction::Up => n,
            Direction::Down => -n,
            Direction::Left => -1,
            Direction::Right => 1,
        }
    }

    /// Const variant of [`offset`] that takes board size explicitly.
    #[must_use]
    pub const fn offset_const(self, n: i8) -> i8 {
        match self {
            Direction::Up => n,
            Direction::Down => -n,
            Direction::Left => -1,
            Direction::Right => 1,
        }
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Up => f.write_char('+'),
            Direction::Down => f.write_char('-'),
            Direction::Left => f.write_char('<'),
            Direction::Right => f.write_char('>'),
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct Square(u8);

impl Square {
    /// Compile-time upper bound on the number of squares (7x7 = 49).
    /// Used for sizing arrays. Use [`Square::count`] for the actual
    /// number of squares for the currently-active board size.
    pub const MAX_COUNT: usize = MAX_SQ;

    /// Number of squares for the currently-active board size (N*N).
    #[must_use]
    #[inline]
    pub fn count() -> usize {
        current_size_sq()
    }

    /// Default square value, useful as an array-fill placeholder.
    pub const A1: Self = Self(0);

    #[must_use]
    pub const fn from_raw(raw: u8) -> Option<Self> {
        if (raw as usize) < Self::MAX_COUNT {
            Some(Self(raw))
        } else {
            None
        }
    }

    #[must_use]
    pub fn from_file_rank(file: u32, rank: u32) -> Option<Self> {
        let n = current_size() as u32;
        if file >= n || rank >= n {
            None
        } else {
            Some(Self((rank * n + file) as u8))
        }
    }

    /// Const constructor used by per-size compile-time tables; takes the
    /// board size explicitly.
    #[must_use]
    pub const fn from_file_rank_const(file: u32, rank: u32, n: u32) -> Option<Self> {
        if file >= n || rank >= n {
            None
        } else {
            Some(Self((rank * n + file) as u8))
        }
    }

    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[must_use]
    pub fn rank(self) -> u32 {
        self.0 as u32 / current_size() as u32
    }

    #[must_use]
    pub fn file(self) -> u32 {
        self.0 as u32 % current_size() as u32
    }

    #[must_use]
    pub fn bb(self) -> Bitboard {
        Bitboard::from_raw(1 << self.idx())
    }

    /// Const single-bit bitboard for this square. Doesn't apply the
    /// runtime size mask, so callers must keep the index in range.
    #[must_use]
    pub const fn bb_const(self) -> Bitboard {
        Bitboard::from_raw_unmasked(1 << self.idx())
    }

    #[must_use]
    pub fn shift(self, dir: Direction) -> Option<Self> {
        let shifted = self.0 as i8 + dir.offset();
        if shifted >= 0 && (shifted as usize) < Self::count() {
            Some(Self(shifted as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub fn shift_checked(self, dir: Direction) -> Option<Self> {
        let n = current_size() as u32;
        match dir {
            Direction::Left if self.file() == 0 => None,
            Direction::Right if self.file() + 1 == n => None,
            _ => self.shift(dir),
        }
    }

    /// Const variant of [`shift_checked`] that takes board size explicitly.
    /// Used by per-size compile-time tables.
    #[must_use]
    pub const fn shift_checked_const(self, dir: Direction, n: u32) -> Option<Self> {
        let file = self.0 as u32 % n;
        match dir {
            Direction::Left => {
                if file == 0 {
                    return None;
                }
            }
            Direction::Right => {
                if file + 1 == n {
                    return None;
                }
            }
            _ => {}
        }
        let off: i8 = match dir {
            Direction::Up => n as i8,
            Direction::Down => -(n as i8),
            Direction::Left => -1,
            Direction::Right => 1,
        };
        let shifted = self.0 as i32 + off as i32;
        if shifted >= 0 && (shifted as u32) < n * n {
            Some(Self(shifted as u8))
        } else {
            None
        }
    }

    #[must_use]
    pub fn all() -> SquareIterator {
        SquareIterator {
            raw: 0,
            limit: Self::count() as u8,
        }
    }
}

impl Display for Square {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char((b'a' + self.file() as u8) as char)?;
        f.write_char((b'1' + self.rank() as u8) as char)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum SquareStrError {
    NonAsciiString,
    WrongLength,
    InvalidFile,
    InvalidRank,
}

impl FromStr for Square {
    type Err = SquareStrError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if !s.is_ascii() {
            return Err(SquareStrError::NonAsciiString);
        }

        let bytes = s.as_bytes();

        if bytes.len() != 2 {
            return Err(SquareStrError::WrongLength);
        }

        let n = current_size();
        let max_file = b'a' + n - 1;
        let max_rank = b'1' + n - 1;

        let file = bytes[0];
        if !(b'a'..=max_file).contains(&file) {
            return Err(SquareStrError::InvalidFile);
        }

        let rank = bytes[1];
        if !(b'1'..=max_rank).contains(&rank) {
            return Err(SquareStrError::InvalidRank);
        }

        Ok(Self::from_file_rank((file - b'a') as u32, (rank - b'1') as u32).unwrap())
    }
}

pub struct SquareIterator {
    raw: u8,
    limit: u8,
}

impl Iterator for SquareIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.raw >= self.limit {
            return None;
        }
        let sq = Square(self.raw);
        self.raw += 1;
        Some(sq)
    }
}
