use core::fmt;

use crate::point::Point;
use crate::symmetry::Symmetry;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum Move<const ROWS: usize, const COLS: usize> {
    Pass,
    Stone(Point<ROWS, COLS>),
}

impl<const ROWS: usize, const COLS: usize> Move<ROWS, COLS> {
    /// Pack into a single integer in `0..=ROWS*COLS`. Stone moves use their
    /// point index, and `Pass` uses `ROWS*COLS`.
    #[inline]
    #[must_use]
    pub const fn encode(self) -> usize {
        match self {
            Self::Stone(p) => p.index(),
            Self::Pass => ROWS * COLS,
        }
    }

    #[inline]
    #[must_use]
    pub const fn decode(value: usize) -> Option<Self> {
        if value == ROWS * COLS {
            Some(Self::Pass)
        } else {
            match Point::<ROWS, COLS>::from_index(value) {
                Some(p) => Some(Self::Stone(p)),
                None => None,
            }
        }
    }

    /// Apply a symmetry transform to a move. `Pass` is invariant.
    #[inline]
    #[must_use]
    pub const fn transform_under(self, sym: Symmetry) -> Self {
        match self {
            Self::Pass => Self::Pass,
            Self::Stone(p) => {
                let new_idx = sym.apply_index(p.index(), ROWS, COLS);
                let new_pt = Point::<ROWS, COLS>::from_index_unchecked(new_idx);
                Self::Stone(new_pt)
            }
        }
    }
}

impl<const ROWS: usize, const COLS: usize> fmt::Debug for Move<ROWS, COLS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pass => f.write_str("Pass"),
            Self::Stone(p) => write!(f, "Stone({}, {})", p.row(), p.col()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_is_invariant_under_every_symmetry() {
        for &sym in Symmetry::all_for(3, 3) {
            assert_eq!(Move::<3, 3>::Pass.transform_under(sym), Move::Pass);
        }
    }

    #[test]
    fn transform_then_inverse_is_identity() {
        let stone = Move::Stone(Point::<3, 3>::new(0, 1).unwrap());
        for &sym in Symmetry::all_for(3, 3) {
            let transformed = stone.transform_under(sym);
            let back = transformed.transform_under(sym.inverse());
            assert_eq!(back, stone);
        }
    }

    #[test]
    fn flip_h_swaps_left_and_right() {
        let left = Move::Stone(Point::<3, 3>::new(1, 0).unwrap());
        let right = Move::Stone(Point::<3, 3>::new(1, 2).unwrap());
        assert_eq!(left.transform_under(Symmetry::FlipH), right);
        assert_eq!(right.transform_under(Symmetry::FlipH), left);
    }
}
