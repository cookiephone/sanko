//! PSK history for the current search path.
//!
//! Holds the set of board keys (`Board::key`, 128 bits, exact on boards of up
//! to 49 points) seen along the path from the root to the current position.
//! `Position::play` pushes the new key before returning, and `Position::unmake`
//! pops it. The set is consulted as the PSK legality check: any move that would
//! create a board already in the set is rejected with
//! [`crate::PlayError::PositionalSuperko`].
//!
//! Implementation: a `Vec<u128>` for ordered push/pop plus a 1024-bucket
//! counting filter (bucket = high bits of the key's low half, the 64-bit board
//! hash): `contains` scans the stack only
//! when the key's bucket is occupied, so push/pop are two array updates and a
//! miss is one load. Paths are short (typically under 100 boards, at most 1,000
//! in the search), so a filter hit's scan is cheap. Both move together. Never
//! edit one without the other.
//!
//! The history is a separate type, not a field of `Position`: transposition
//! table keys are board-state-only (`board hash ^ STM`). Embedding history in
//! `Position` would make TT lookups history-sensitive and defeat their
//! purpose. GHI handling lives in the search layer's TT.

const FILTER_BITS: u32 = 10;

#[inline]
const fn bucket(key: u128) -> usize {
    ((key as u64) >> (64 - FILTER_BITS)) as usize
}

#[derive(Clone, Debug)]
pub struct HashHistory {
    stack: Vec<u128>,
    /// Number of stacked keys per bucket.
    counts: [u16; 1 << FILTER_BITS],
}

impl Default for HashHistory {
    fn default() -> Self {
        Self { stack: Vec::new(), counts: [0; 1 << FILTER_BITS] }
    }
}

impl HashHistory {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-allocate for an expected path length.
    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            stack: Vec::with_capacity(cap),
            counts: [0; 1 << FILTER_BITS],
        }
    }

    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.stack.len()
    }

    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    /// `true` if any prior board on the path had this key.
    #[inline]
    #[must_use]
    pub fn contains(&self, key: u128) -> bool {
        self.counts[bucket(key)] != 0 && self.stack.contains(&key)
    }

    /// Push the key of a board that is now part of the path. No check is made
    /// here: the caller tests `contains` first (positional superko). A
    /// duplicate would still be stacked and counted, so `pop` stays consistent.
    pub fn push(&mut self, key: u128) {
        self.stack.push(key);
        self.counts[bucket(key)] += 1;
    }

    /// Remove the most recent push and return it, or `None` if the path is empty.
    pub fn pop(&mut self) -> Option<u128> {
        let h = self.stack.pop()?;
        // The bucket count tracks multiplicity and `contains` rescans the
        // stack, so this is correct even if the hash also occurs earlier.
        // Under PSK it never does: the caller rejects a repeat before the push.
        self.counts[bucket(h)] -= 1;
        Some(h)
    }

    pub fn clear(&mut self) {
        self.stack.clear();
        self.counts = [0; 1 << FILTER_BITS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_roundtrip() {
        let mut h = HashHistory::new();
        h.push(1);
        h.push(2);
        h.push(3);
        assert!(h.contains(2));
        assert_eq!(h.len(), 3);
        assert_eq!(h.pop(), Some(3));
        assert!(!h.contains(3));
        assert_eq!(h.pop(), Some(2));
        assert_eq!(h.pop(), Some(1));
        assert!(h.is_empty());
    }

    #[test]
    fn contains_after_pop_is_false() {
        let mut h = HashHistory::new();
        h.push(42);
        assert!(h.contains(42));
        h.pop();
        assert!(!h.contains(42));
    }
}
