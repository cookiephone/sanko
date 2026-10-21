use core::fmt;

use crate::bitboard::BitBoard;
use crate::color::Color;
use crate::point::Point;
use crate::symmetry::Symmetry;
use crate::zobrist::Zobrist;

/// A Go board with `ROWS × COLS` points (rectangular, and square is just the
/// `ROWS == COLS` case).
///
/// `LANES` and `NPOINTS` are mechanical functions of `ROWS` and `COLS` that
/// the type system cannot yet derive on stable Rust. Use the `BoardRxC`
/// type aliases or the `define_board!` macro rather than naming this type
/// directly.
///
/// `Board` enforces nothing about Go rules: no captures, no ko, no suicide
/// check. The rules layer (`rules.rs`, `position.rs`) is responsible for all
/// of that.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Board<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> {
    black: BitBoard<LANES>,
    white: BitBoard<LANES>,
    hash: u64,
    /// The board's hash under the second, independent table (see [`Self::key`]).
    hash2: u64,
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>
    Board<ROWS, COLS, LANES, NPOINTS>
{
    pub(crate) const ZOBRIST: Zobrist<NPOINTS> = Zobrist::new();
    pub(crate) const ZOBRIST2: Zobrist<NPOINTS> = Zobrist::second();

    /// Per-symmetry Zobrist keys, precomputed at compile time:
    /// `SYM_ZOBRIST[sym as usize][idx][color] == ZOBRIST.key(sym.apply_index(idx), color)`.
    /// Lets [`Self::hash_under_symmetry`] reduce to one XOR per set bit (no
    /// `apply_rc` div/mod/branch). Square-only symmetries (indices ≥ 4) are left
    /// zero on rectangular boards. They are never used there (`Symmetry::all_for`
    /// returns only the Klein-four group), and computing them would index the
    /// Zobrist table out of range.
    const SYM_ZOBRIST: [[[u64; 2]; NPOINTS]; 8] = Self::sym_table(&Self::ZOBRIST);
    /// The same for the second table.
    const SYM_ZOBRIST2: [[[u64; 2]; NPOINTS]; 8] = Self::sym_table(&Self::ZOBRIST2);

    const fn sym_table(z: &Zobrist<NPOINTS>) -> [[[u64; 2]; NPOINTS]; 8] {
        let syms = [
            Symmetry::Identity,
            Symmetry::FlipH,
            Symmetry::FlipV,
            Symmetry::Rot180,
            Symmetry::Rot90,
            Symmetry::Rot270,
            Symmetry::DiagMain,
            Symmetry::DiagAnti,
        ];
        let mut table = [[[0u64; 2]; NPOINTS]; 8];
        let mut s = 0;
        while s < 8 {
            // Square-only symmetries are valid only when ROWS == COLS.
            if s < 4 || ROWS == COLS {
                let mut idx = 0;
                while idx < NPOINTS {
                    let dst = syms[s].apply_index(idx, ROWS, COLS);
                    table[s][idx][0] = z.key(dst, Color::Black);
                    table[s][idx][1] = z.key(dst, Color::White);
                    idx += 1;
                }
            }
            s += 1;
        }
        table
    }

    #[must_use]
    pub const fn new() -> Self {
        const {
            assert!(ROWS >= 1, "ROWS must be at least 1");
            assert!(COLS >= 1, "COLS must be at least 1");
            assert!(NPOINTS == ROWS * COLS, "NPOINTS must equal ROWS * COLS");
            assert!(
                LANES == NPOINTS.div_ceil(64),
                "LANES must be ceil(NPOINTS / 64)"
            );
        }
        Self {
            black: BitBoard::EMPTY,
            white: BitBoard::EMPTY,
            hash: 0,
            hash2: 0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn rows() -> usize {
        ROWS
    }

    #[inline]
    #[must_use]
    pub const fn cols() -> usize {
        COLS
    }

    #[inline]
    #[must_use]
    pub const fn npoints() -> usize {
        NPOINTS
    }

    /// Position-only Zobrist hash (no STM, no ko, no history).
    #[inline]
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Position-only hash under the second, independent table.
    #[inline]
    #[must_use]
    pub const fn hash2(&self) -> u64 {
        self.hash2
    }

    /// 128-bit board key: [`Self::hash`] in the low half, [`Self::hash2`] in the
    /// high half. It is exact (distinct boards have distinct keys) on boards of
    /// up to 49 points, where the 2N key pairs are linearly independent over
    /// GF(2) (tested in `zobrist.rs`). Up to 30 points the high half is 0 and the
    /// 64-bit hash alone is exact.
    #[inline]
    #[must_use]
    pub fn key(&self) -> u128 {
        (u128::from(self.hash2) << 64) | u128::from(self.hash)
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.black.is_empty() && self.white.is_empty()
    }

    #[inline]
    #[must_use]
    pub const fn stone_count(&self) -> u32 {
        self.black.count() + self.white.count()
    }

    #[inline]
    #[must_use]
    pub const fn stones(&self, color: Color) -> BitBoard<LANES> {
        match color {
            Color::Black => self.black,
            Color::White => self.white,
        }
    }

    /// All empty points. Computed as the complement of `black ∪ white`,
    /// masked to `NPOINTS`.
    #[inline]
    #[must_use]
    pub const fn empties(&self) -> BitBoard<LANES> {
        let mut mask = BitBoard::<LANES>::EMPTY;
        let mut i = 0;
        while i < NPOINTS {
            mask.set(i);
            i += 1;
        }
        let occupied = self.black.union(&self.white);
        mask.difference(&occupied)
    }

    #[inline]
    #[must_use]
    pub const fn at(&self, point: Point<ROWS, COLS>) -> Option<Color> {
        let idx = point.index();
        if self.black.get(idx) {
            Some(Color::Black)
        } else if self.white.get(idx) {
            Some(Color::White)
        } else {
            None
        }
    }

    /// Hash of the board as if every stone had been transformed by `sym`.
    /// Used by the search layer's symmetry-aware TT to canonicalize positions.
    #[must_use]
    pub fn hash_under_symmetry(&self, sym: Symmetry) -> u64 {
        if matches!(sym, Symmetry::Identity) {
            return self.hash;
        }
        // Iterate only occupied points (set bits) and fold in the precomputed
        // per-symmetry Zobrist key (no per-cell `at()` probe, no `apply_rc`
        // div/mod/branch). The searcher calls this once per symmetry when it
        // starts and then updates the symmetric hashes incrementally per move.
        let table = &Self::SYM_ZOBRIST[sym as usize];
        let mut h = 0u64;
        for idx in &self.black {
            h ^= table[idx][0];
        }
        for idx in &self.white {
            h ^= table[idx][1];
        }
        h
    }

    /// [`Self::hash_under_symmetry`] for the second table. Zero up to 30 points.
    #[must_use]
    pub fn hash2_under_symmetry(&self, sym: Symmetry) -> u64 {
        if !Self::WIDE {
            return 0;
        }
        let table = &Self::SYM_ZOBRIST2[sym as usize];
        let mut h = 0u64;
        for idx in &self.black {
            h ^= table[idx][0];
        }
        for idx in &self.white {
            h ^= table[idx][1];
        }
        h
    }

    /// Zobrist key of a `color` stone at `idx` *as seen under* symmetry number
    /// `sym` (the index into `Symmetry::all_for`'s square ordering). Lets a
    /// search keep all (up to eight) symmetric board hashes incrementally: XOR this key
    /// in/out for each placed/removed stone instead of re-folding the board.
    #[inline]
    #[must_use]
    pub const fn sym_zobrist(sym: usize, idx: usize, color: Color) -> u64 {
        Self::SYM_ZOBRIST[sym][idx][color as usize]
    }

    /// [`Self::sym_zobrist`] for the second table.
    #[inline]
    #[must_use]
    pub const fn sym_zobrist2(sym: usize, idx: usize, color: Color) -> u64 {
        Self::SYM_ZOBRIST2[sym][idx][color as usize]
    }

    /// Boards above 30 points keep the second hash: there the 64-bit hash is
    /// not exact, and [`Self::key`] needs both halves. Up to 30 points the
    /// second hash stays 0, so smaller boards run exactly as before.
    pub const WIDE: bool = NPOINTS > 30;

    /// Place a stone unconditionally and update the hash. Caller must ensure
    /// the point is empty.
    #[inline]
    pub(crate) fn place(&mut self, point: Point<ROWS, COLS>, color: Color) {
        debug_assert!(self.at(point).is_none(), "place() onto an occupied point");
        let idx = point.index();
        match color {
            Color::Black => self.black.set(idx),
            Color::White => self.white.set(idx),
        }
        self.hash ^= Self::ZOBRIST.key(idx, color);
        if Self::WIDE {
            self.hash2 ^= Self::ZOBRIST2.key(idx, color);
        }
    }

    /// Remove a stone unconditionally and update the hash.
    #[inline]
    pub(crate) fn remove(&mut self, point: Point<ROWS, COLS>, color: Color) {
        debug_assert!(self.at(point) == Some(color), "remove() with wrong color");
        let idx = point.index();
        match color {
            Color::Black => self.black.clear(idx),
            Color::White => self.white.clear(idx),
        }
        self.hash ^= Self::ZOBRIST.key(idx, color);
        if Self::WIDE {
            self.hash2 ^= Self::ZOBRIST2.key(idx, color);
        }
    }
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> Default
    for Board<ROWS, COLS, LANES, NPOINTS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> fmt::Display
    for Board<ROWS, COLS, LANES, NPOINTS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..ROWS {
            for col in 0..COLS {
                let pt = Point::<ROWS, COLS>::new_unchecked(row, col);
                let ch = match self.at(pt) {
                    None => '.',
                    Some(Color::Black) => 'X',
                    Some(Color::White) => 'O',
                };
                if col > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "{ch}")?;
            }
            if row + 1 < ROWS {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> fmt::Debug
    for Board<ROWS, COLS, LANES, NPOINTS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Board<{ROWS}x{COLS}> hash={:#018x}", self.hash)?;
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::define_board;

    define_board!(B5x5, 5, 5);
    define_board!(B4x5, 4, 5);

    #[test]
    fn empty_board_has_zero_hash() {
        let b = B5x5::new();
        assert_eq!(b.hash(), 0);
        assert!(b.is_empty());
        assert_eq!(b.stone_count(), 0);
    }

    #[test]
    fn place_then_remove_returns_to_empty_hash() {
        let mut b = B5x5::new();
        let p = Point::<5, 5>::new(2, 2).unwrap();
        b.place(p, Color::Black);
        assert!(!b.is_empty());
        assert_ne!(b.hash(), 0);
        b.remove(p, Color::Black);
        assert!(b.is_empty());
        assert_eq!(b.hash(), 0);
    }

    #[test]
    fn place_order_does_not_affect_hash() {
        let p1 = Point::<5, 5>::new(0, 0).unwrap();
        let p2 = Point::<5, 5>::new(2, 3).unwrap();
        let p3 = Point::<5, 5>::new(4, 4).unwrap();

        let mut a = B5x5::new();
        a.place(p1, Color::Black);
        a.place(p2, Color::White);
        a.place(p3, Color::Black);

        let mut b = B5x5::new();
        b.place(p3, Color::Black);
        b.place(p1, Color::Black);
        b.place(p2, Color::White);

        assert_eq!(a.hash(), b.hash());
        assert_eq!(a, b);
    }

    #[test]
    fn rectangular_display() {
        let mut b = B4x5::new();
        b.place(Point::<4, 5>::new(0, 0).unwrap(), Color::Black);
        b.place(Point::<4, 5>::new(3, 4).unwrap(), Color::White);
        let s = format!("{b}");
        assert_eq!(s, "X . . . .\n. . . . .\n. . . . .\n. . . . O");
    }
}
