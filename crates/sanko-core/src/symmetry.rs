//! Board symmetries.
//!
//! Rectangular boards have a Klein-four symmetry group:
//! `{identity, h-flip, v-flip, 180°}`. Square boards add the 90° and 270°
//! rotations plus the two diagonal reflections, giving the full D4 group of
//! order 8.

extern crate alloc;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub enum Symmetry {
    Identity,
    FlipH,
    FlipV,
    Rot180,
    /// 90° CCW rotation. Square boards only.
    Rot90,
    /// 90° CW rotation. Square boards only.
    Rot270,
    /// Main-diagonal reflection. Square boards only.
    DiagMain,
    /// Anti-diagonal reflection. Square boards only.
    DiagAnti,
}

impl Symmetry {
    #[inline]
    #[must_use]
    pub const fn apply_rc(self, r: usize, c: usize, rows: usize, cols: usize) -> (usize, usize) {
        match self {
            Self::Identity => (r, c),
            Self::FlipH => (r, cols - 1 - c),
            Self::FlipV => (rows - 1 - r, c),
            Self::Rot180 => (rows - 1 - r, cols - 1 - c),
            Self::Rot90 => (cols - 1 - c, r),
            Self::Rot270 => (c, rows - 1 - r),
            Self::DiagMain => (c, r),
            Self::DiagAnti => (cols - 1 - c, rows - 1 - r),
        }
    }

    #[inline]
    #[must_use]
    pub const fn apply_index(self, idx: usize, rows: usize, cols: usize) -> usize {
        let r = idx / cols;
        let c = idx % cols;
        let (r2, c2) = self.apply_rc(r, c, rows, cols);
        r2 * cols + c2
    }

    #[inline]
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Rot90 => Self::Rot270,
            Self::Rot270 => Self::Rot90,
            other => other,
        }
    }

    /// Every symmetry valid on a `ROWS × COLS` board.
    ///
    /// Square boards get full D4 (8 elements), rectangular ones Klein four (4).
    #[must_use]
    pub const fn all_for(rows: usize, cols: usize) -> &'static [Self] {
        const RECTANGULAR: [Symmetry; 4] = [
            Symmetry::Identity,
            Symmetry::FlipH,
            Symmetry::FlipV,
            Symmetry::Rot180,
        ];
        const SQUARE: [Symmetry; 8] = [
            Symmetry::Identity,
            Symmetry::FlipH,
            Symmetry::FlipV,
            Symmetry::Rot180,
            Symmetry::Rot90,
            Symmetry::Rot270,
            Symmetry::DiagMain,
            Symmetry::DiagAnti,
        ];
        if rows == cols { &SQUARE } else { &RECTANGULAR }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_identity() {
        for r in 0..5 {
            for c in 0..5 {
                assert_eq!(Symmetry::Identity.apply_rc(r, c, 5, 5), (r, c));
            }
        }
    }

    #[test]
    fn flips_are_involutory() {
        for sym in [Symmetry::FlipH, Symmetry::FlipV, Symmetry::Rot180] {
            for r in 0..4 {
                for c in 0..5 {
                    let (r2, c2) = sym.apply_rc(r, c, 4, 5);
                    let (r3, c3) = sym.apply_rc(r2, c2, 4, 5);
                    assert_eq!((r3, c3), (r, c));
                }
            }
        }
    }

    #[test]
    fn rotate_90_inverse_is_rotate_270() {
        for r in 0..5 {
            for c in 0..5 {
                let (r2, c2) = Symmetry::Rot90.apply_rc(r, c, 5, 5);
                let (r3, c3) = Symmetry::Rot270.apply_rc(r2, c2, 5, 5);
                assert_eq!((r3, c3), (r, c));
            }
        }
    }

    #[test]
    fn rectangular_has_four_symmetries_square_has_eight() {
        assert_eq!(Symmetry::all_for(3, 5).len(), 4);
        assert_eq!(Symmetry::all_for(5, 5).len(), 8);
    }
}
