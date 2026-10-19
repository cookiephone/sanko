//! Shared 4-connected dilation on `BitBoard<LANES>`. Kernel of every
//! flood-fill in the engine (chain liberty counts, area scoring).
//!
//! For `LANES = 1` (every board up to 8×8 = 64 points) this is a straight-line
//! shift+mask sequence: a handful of 64-bit ops, no branches, no table lookups.

use crate::bitboard::BitBoard;

/// `b ∪ {orthogonal neighbors of every bit of b}` under 4-connectivity,
/// respecting board bounds.
#[inline]
#[must_use]
pub(crate) fn dilate<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
>(
    b: &BitBoard<LANES>,
) -> BitBoard<LANES> {
    if LANES == 1 {
        let v = b.lane(0);
        let north = v << COLS;
        let south = v >> COLS;
        let east = (v & not_col_mask::<COLS>(COLS - 1)) << 1;
        let west = (v & not_col_mask::<COLS>(0)) >> 1;
        let board_mask: u64 = if NPOINTS >= 64 {
            !0u64
        } else {
            (1u64 << NPOINTS) - 1
        };
        let out = (v | north | south | east | west) & board_mask;
        BitBoard::from_lane0(out)
    } else {
        let mut out = *b;
        for idx in b {
            let pt = crate::Point::<ROWS, COLS>::from_index_unchecked(idx);
            for nb in pt.neighbors().into_iter().flatten() {
                out.set(nb.index());
            }
        }
        out
    }
}

/// Orthogonal neighbours of a ≤64-point board's bits (without the bits
/// themselves, unless one is adjacent to another), within the board.
#[inline]
#[must_use]
pub(crate) const fn neighbors_bits<const COLS: usize, const NPOINTS: usize>(v: u64) -> u64 {
    let north = v << COLS;
    let south = v >> COLS;
    let east = (v & not_col_mask::<COLS>(COLS - 1)) << 1;
    let west = (v & not_col_mask::<COLS>(0)) >> 1;
    let board_mask: u64 = if NPOINTS >= 64 { !0u64 } else { (1u64 << NPOINTS) - 1 };
    (north | south | east | west) & board_mask
}

#[inline]
const fn not_col_mask<const COLS: usize>(col: usize) -> u64 {
    let mut m: u64 = 0;
    let mut r = 0usize;
    while r * COLS + col < 64 {
        m |= 1u64 << (r * COLS + col);
        r += 1;
    }
    !m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_dilate<const ROWS: usize, const COLS: usize, const LANES: usize>(
        b: &BitBoard<LANES>,
    ) -> BitBoard<LANES> {
        let mut out = *b;
        for idx in b {
            let pt = crate::Point::<ROWS, COLS>::from_index_unchecked(idx);
            for nb in pt.neighbors().into_iter().flatten() {
                out.set(nb.index());
            }
        }
        out
    }

    #[test]
    fn dilate_empty_is_empty() {
        let b: BitBoard<1> = BitBoard::EMPTY;
        let d = dilate::<5, 5, 1, 25>(&b);
        assert!(d.is_empty());
    }

    #[test]
    fn dilate_single_center_5x5() {
        let mut b: BitBoard<1> = BitBoard::EMPTY;
        b.set(12);
        let d = dilate::<5, 5, 1, 25>(&b);
        let r = reference_dilate::<5, 5, 1>(&b);
        assert_eq!(d, r);
        let expected: u64 = (1 << 7) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 17);
        assert_eq!(d.lane(0), expected);
    }

    #[test]
    fn dilate_corner_no_wrap_5x5() {
        let mut b: BitBoard<1> = BitBoard::EMPTY;
        b.set(0);
        let d = dilate::<5, 5, 1, 25>(&b);
        let r = reference_dilate::<5, 5, 1>(&b);
        assert_eq!(d, r);
        let expected: u64 = 1 | (1 << 1) | (1 << 5);
        assert_eq!(d.lane(0), expected);
    }

    #[test]
    fn dilate_right_edge_does_not_wrap_5x5() {
        let mut b: BitBoard<1> = BitBoard::EMPTY;
        b.set(4);
        let d = dilate::<5, 5, 1, 25>(&b);
        let r = reference_dilate::<5, 5, 1>(&b);
        assert_eq!(d, r);
        let expected: u64 = (1 << 3) | (1 << 4) | (1 << 9);
        assert_eq!(d.lane(0), expected);
    }

    #[test]
    fn dilate_left_edge_does_not_wrap_5x5() {
        let mut b: BitBoard<1> = BitBoard::EMPTY;
        b.set(5);
        let d = dilate::<5, 5, 1, 25>(&b);
        let r = reference_dilate::<5, 5, 1>(&b);
        assert_eq!(d, r);
        let expected: u64 = 1 | (1 << 5) | (1 << 6) | (1 << 10);
        assert_eq!(d.lane(0), expected);
    }

    #[test]
    fn dilate_bottom_row_no_overflow_7x7() {
        let mut b: BitBoard<1> = BitBoard::EMPTY;
        b.set(48);
        let d = dilate::<7, 7, 1, 49>(&b);
        let r = reference_dilate::<7, 7, 1>(&b);
        assert_eq!(d, r);
        assert_eq!(d.lane(0) & !((1u64 << 49) - 1), 0);
    }
}
