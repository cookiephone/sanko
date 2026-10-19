use core::fmt;

/// A point on a `ROWS × COLS` board, stored as a flat index in `0..ROWS*COLS`.
/// Row-major: `index = row * COLS + col`.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct Point<const ROWS: usize, const COLS: usize> {
    idx: usize,
}

impl<const ROWS: usize, const COLS: usize> Point<ROWS, COLS> {
    pub const NPOINTS: usize = ROWS * COLS;
    const NEIGHBORS_TABLE_NESTED: [[[Option<Self>; 4]; COLS]; ROWS] = {
        let mut table = [[[None; 4]; COLS]; ROWS];
        let mut r = 0;
        while r < ROWS {
            let mut c = 0;
            while c < COLS {
                table[r][c] = [
                    if r > 0 { Self::new(r - 1, c) } else { None },
                    if r + 1 < ROWS {
                        Self::new(r + 1, c)
                    } else {
                        None
                    },
                    if c > 0 { Self::new(r, c - 1) } else { None },
                    if c + 1 < COLS {
                        Self::new(r, c + 1)
                    } else {
                        None
                    },
                ];
                c += 1;
            }
            r += 1;
        }
        table
    };
    const NEIGHBORS_TABLE: &'static [[Option<Self>; 4]] =
        Self::NEIGHBORS_TABLE_NESTED.as_flattened();

    #[inline]
    #[must_use]
    pub const fn new(row: usize, col: usize) -> Option<Self> {
        if row < ROWS && col < COLS {
            Some(Self {
                idx: row * COLS + col,
            })
        } else {
            None
        }
    }

    #[inline]
    #[must_use]
    pub const fn new_unchecked(row: usize, col: usize) -> Self {
        Self {
            idx: row * COLS + col,
        }
    }

    #[inline]
    #[must_use]
    pub const fn from_index(idx: usize) -> Option<Self> {
        if idx < Self::NPOINTS {
            Some(Self { idx })
        } else {
            None
        }
    }

    #[inline]
    #[must_use]
    pub const fn from_index_unchecked(idx: usize) -> Self {
        Self { idx }
    }

    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.idx
    }

    #[inline]
    #[must_use]
    pub const fn row(self) -> usize {
        self.idx / COLS
    }

    #[inline]
    #[must_use]
    pub const fn col(self) -> usize {
        self.idx % COLS
    }

    /// Up to four orthogonal neighbors, in the order
    /// (up, down, left, right). `None` entries are off-board sides. Callers
    /// flatten with `.into_iter().flatten()` to skip them.
    #[inline]
    #[must_use]
    pub const fn neighbors(self) -> [Option<Self>; 4] {
        Self::NEIGHBORS_TABLE[self.idx]
    }
}

impl<const ROWS: usize, const COLS: usize> fmt::Debug for Point<ROWS, COLS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Point<{}x{}>(r={}, c={})",
            ROWS,
            COLS,
            self.row(),
            self.col()
        )
    }
}
