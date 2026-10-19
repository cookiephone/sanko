//! Packed bitset over `LANES` × 64 bits.
//!
//! Public type, exposed so higher-level analysis can return bitboards directly
//! without materializing into `Vec<Point>`. Mutators are crate-private, so the
//! type is read-only from outside the core crate.

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BitBoard<const LANES: usize> {
    lanes: [u64; LANES],
}

impl<const LANES: usize> Default for BitBoard<LANES> {
    #[inline]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<const LANES: usize> BitBoard<LANES> {
    pub const EMPTY: Self = Self { lanes: [0; LANES] };

    /// All-ones over the first `npoints` bits (the on-board points), zeros
    /// elsewhere. `npoints` must be `≤ LANES * 64`.
    #[inline]
    #[must_use]
    pub const fn full(npoints: usize) -> Self {
        let mut b = Self::EMPTY;
        let mut i = 0;
        while i < npoints {
            b.set(i);
            i += 1;
        }
        b
    }

    /// A bitboard with exactly the bit `idx` set.
    #[inline]
    #[must_use]
    pub const fn singleton(idx: usize) -> Self {
        let mut b = Self::EMPTY;
        b.set(idx);
        b
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        let mut i = 0;
        while i < LANES {
            if self.lanes[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline]
    #[must_use]
    pub const fn count(&self) -> u32 {
        let mut total = 0u32;
        let mut i = 0;
        while i < LANES {
            total += self.lanes[i].count_ones();
            i += 1;
        }
        total
    }

    #[inline]
    #[must_use]
    pub const fn get(&self, idx: usize) -> bool {
        let lane = idx / 64;
        let bit = idx % 64;
        (self.lanes[lane] >> bit) & 1 != 0
    }

    #[inline]
    pub(crate) const fn set(&mut self, idx: usize) {
        let lane = idx / 64;
        let bit = idx % 64;
        self.lanes[lane] |= 1u64 << bit;
    }

    #[inline]
    pub(crate) const fn clear(&mut self, idx: usize) {
        let lane = idx / 64;
        let bit = idx % 64;
        self.lanes[lane] &= !(1u64 << bit);
    }

    /// The first 64 bits (all of the board for boards up to 64 points).
    #[inline]
    #[must_use]
    pub const fn bits(&self) -> u64 {
        self.lanes[0]
    }

    #[inline]
    #[must_use]
    pub(crate) const fn lane(&self, i: usize) -> u64 {
        self.lanes[i]
    }

    #[inline]
    #[must_use]
    pub(crate) const fn from_lane0(v: u64) -> Self {
        let mut lanes = [0u64; LANES];
        lanes[0] = v;
        Self { lanes }
    }

    #[inline]
    #[must_use]
    pub const fn union(&self, other: &Self) -> Self {
        let mut out = [0u64; LANES];
        let mut i = 0;
        while i < LANES {
            out[i] = self.lanes[i] | other.lanes[i];
            i += 1;
        }
        Self { lanes: out }
    }

    #[inline]
    #[must_use]
    pub const fn intersect(&self, other: &Self) -> Self {
        let mut out = [0u64; LANES];
        let mut i = 0;
        while i < LANES {
            out[i] = self.lanes[i] & other.lanes[i];
            i += 1;
        }
        Self { lanes: out }
    }

    /// Bitwise AND-NOT (`self & !other`).
    #[inline]
    #[must_use]
    pub const fn difference(&self, other: &Self) -> Self {
        let mut out = [0u64; LANES];
        let mut i = 0;
        while i < LANES {
            out[i] = self.lanes[i] & !other.lanes[i];
            i += 1;
        }
        Self { lanes: out }
    }

    #[inline]
    #[must_use]
    pub const fn is_subset_of(&self, other: &Self) -> bool {
        let mut i = 0;
        while i < LANES {
            if self.lanes[i] & !other.lanes[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Iterate set bit indices in ascending order, allocation-free.
    #[inline]
    #[must_use]
    pub const fn iter(&self) -> Iter<LANES> {
        Iter {
            lanes: self.lanes,
            cursor: 0,
        }
    }
}

impl<const LANES: usize> IntoIterator for &BitBoard<LANES> {
    type Item = usize;
    type IntoIter = Iter<LANES>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<const LANES: usize> core::fmt::Debug for BitBoard<LANES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bits: alloc::vec::Vec<usize> = self.iter().collect();
        write!(f, "BitBoard<{LANES}>({bits:?})")
    }
}

extern crate alloc;

#[derive(Clone, Debug)]
pub struct Iter<const LANES: usize> {
    lanes: [u64; LANES],
    cursor: usize,
}

impl<const LANES: usize> Iterator for Iter<LANES> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.cursor < LANES {
            let lane = self.lanes[self.cursor];
            if lane != 0 {
                let bit = lane.trailing_zeros() as usize;
                self.lanes[self.cursor] = lane & (lane - 1);
                return Some(self.cursor * 64 + bit);
            }
            self.cursor += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_bits() {
        let b: BitBoard<2> = BitBoard::EMPTY;
        assert!(b.is_empty());
        assert_eq!(b.count(), 0);
    }

    #[test]
    fn set_get_clear_roundtrip() {
        let mut b: BitBoard<2> = BitBoard::EMPTY;
        for &idx in &[0usize, 1, 63, 64, 65, 127] {
            assert!(!b.get(idx));
            b.set(idx);
            assert!(b.get(idx));
            assert_eq!(b.count(), 1);
            b.clear(idx);
            assert!(!b.get(idx));
            assert!(b.is_empty());
        }
    }

    #[test]
    fn iter_yields_set_bits_in_order() {
        let mut b: BitBoard<2> = BitBoard::EMPTY;
        let bits = [3usize, 17, 63, 64, 100, 127];
        for &i in &bits {
            b.set(i);
        }
        let collected: Vec<usize> = b.iter().collect();
        assert_eq!(collected, bits);
    }

    fn make<const L: usize>(bits: &[usize]) -> BitBoard<L> {
        let mut b: BitBoard<L> = BitBoard::EMPTY;
        for &i in bits {
            b.set(i);
        }
        b
    }

    #[test]
    fn union_intersect_difference() {
        let lhs: BitBoard<2> = make(&[1, 5, 64, 100]);
        let rhs: BitBoard<2> = make(&[5, 7, 100, 127]);
        let union = lhs.union(&rhs);
        assert_eq!(
            union.iter().collect::<Vec<_>>(),
            vec![1, 5, 7, 64, 100, 127]
        );
        let intersection = lhs.intersect(&rhs);
        assert_eq!(intersection.iter().collect::<Vec<_>>(), vec![5, 100]);
        let diff = lhs.difference(&rhs);
        assert_eq!(diff.iter().collect::<Vec<_>>(), vec![1, 64]);
    }
}
