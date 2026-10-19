//! Bounded cache of positions whose static-safety verdict (pure-eye analysis)
//! is *undecided*, keyed by canonical position.
//!
//! The static verdict (`sanko_core::statically_decided_life`) runs the pure-eye
//! analysis, and the search invokes it for every child, while df-pn revisits
//! positions many times. The verdict is a pure function of (board, side to
//! move, komi) and symmetry-invariant, so it is keyed on the canonical position
//! key (which includes the side to move and the pass state). Komi is fixed per
//! solve, and each solve has its own cache.
//!
//! Only "undecided" is cached, and the API can record nothing else. A decided
//! verdict becomes a proof leaf. The key would identify it exactly on boards of
//! up to 30 points (see `PnEntry::check`), but keeping decided verdicts out makes
//! the cache's correctness independent of that property: a false "undecided"
//! hit merely forgoes a leaf, and decided positions are rare, so recomputing
//! them is cheap. The history-dependent conditions of a static leaf (no winner
//! stone captured in the territory, and the pass condition) are checked by the
//! caller after the cache, never cached.
//! Direct-mapped, replace-on-collision, exact 64-bit key check, bounded.
//! `log_slots == 0` disables it.

use std::sync::atomic::{AtomicU64, Ordering};

use sanko_core::{Color, Position, statically_decided};

/// Low bits of a slot's word (the other bits hold the key, see
/// [`StaticCache`]): `0` means the slot is empty, `UNDECIDED` that the key's
/// static verdict is undecided.
const UNDECIDED: u64 = 1;

/// A bounded, direct-mapped cache of undecided static-safety verdicts. See
/// module docs.
///
/// One `AtomicU64` per slot: the key with its two low bits replaced by the
/// mark. The slot index is `key & mask` (mask ≥ 3), so every key that lands in
/// a slot shares those two low bits, so comparing the remaining bits is an
/// exact key check, and a probe costs a single cache line.
///
/// ## Concurrency (shared across parallel workers)
///
/// Probe/insert take `&self` and use `Relaxed` atomics on one 64-bit word, so a
/// read is never torn: a hit is always a key that some worker marked
/// undecided. Hit/miss counters are not kept here (they would be a shared
/// write).
#[derive(Debug)]
pub struct StaticCache {
    slots: Vec<AtomicU64>,
    mask: u64,
}

impl StaticCache {
    /// `2^log_slots` slots, disabled when `log_slots == 0`. `log_slots` must be in
    /// `2..=32` when enabled.
    #[must_use]
    pub fn new(log_slots: u32) -> Self {
        let slots = if log_slots == 0 {
            Vec::new()
        } else {
            assert!((2..=32).contains(&log_slots), "log_slots out of range");
            (0..1usize << log_slots).map(|_| AtomicU64::new(0)).collect()
        };
        let mask = (slots.len() as u64).wrapping_sub(1);
        Self { slots, mask }
    }

    #[inline]
    #[must_use]
    pub const fn enabled(&self) -> bool {
        !self.slots.is_empty()
    }

    /// Fraction of slots in use, estimated from up to `samples` slots spread
    /// evenly over the cache. Read-only, for progress displays.
    #[must_use]
    pub fn fill(&self, samples: usize) -> f64 {
        let len = self.slots.len();
        if len == 0 || samples == 0 {
            return 0.0;
        }
        let n = samples.min(len);
        let step = len / n;
        let used = (0..n).filter(|i| self.slots[i * step].load(Ordering::Relaxed) & 3 != 0).count();
        used as f64 / n as f64
    }

    /// Prefetch the slot for `key` (see `PnTable::touch`).
    #[inline]
    pub fn touch(&self, key: u64) {
        if !self.slots.is_empty() {
            crate::pntt::prefetch(&self.slots[(key & self.mask) as usize]);
        }
    }

    /// Whether `key` is cached as undecided. `false` is a miss, not a decided
    /// verdict.
    #[must_use]
    pub fn is_undecided(&self, key: u64) -> bool {
        if self.slots.is_empty() {
            return false;
        }
        let w = self.slots[(key & self.mask) as usize].load(Ordering::Relaxed);
        w & 3 == UNDECIDED && (w ^ key) & !3 == 0
    }

    /// Record that the static verdict for `key` is undecided (overwriting the
    /// slot on collision). No-op when disabled.
    pub fn mark_undecided(&self, key: u64) {
        if self.slots.is_empty() {
            return;
        }
        self.slots[(key & self.mask) as usize].store((key & !3) | UNDECIDED, Ordering::Relaxed);
    }

    /// Cached `statically_decided`: return the verdict for `pos` (whose canonical
    /// `key` the caller already computed), caching it only when undecided (see
    /// module docs). Used for the root, the search's other nodes call
    /// `statically_decided_life` directly.
    #[must_use]
    pub fn verdict<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
        &self,
        key: u64,
        pos: &Position<ROWS, COLS, LANES, NPOINTS>,
        komi_times_two: i32,
    ) -> Option<Color> {
        if self.is_undecided(key) {
            return None;
        }
        let w = statically_decided::<ROWS, COLS, LANES, NPOINTS>(pos, komi_times_two);
        if w.is_none() {
            self.mark_undecided(key);
        }
        w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_and_miss() {
        let c = StaticCache::new(8);
        assert!(!c.is_undecided(5)); // miss
        c.mark_undecided(5);
        c.mark_undecided(6);
        assert!(c.is_undecided(5));
        assert!(c.is_undecided(6));
        assert!(!c.is_undecided(7));
        assert!(!c.is_undecided(5 + 256)); // same slot, different key
        c.mark_undecided(5 + 256); // overwrites the slot
        assert!(c.is_undecided(5 + 256));
        assert!(!c.is_undecided(5));
    }

    #[test]
    fn disabled_always_misses() {
        let c = StaticCache::new(0);
        assert!(!c.enabled());
        c.mark_undecided(1);
        assert!(!c.is_undecided(1));
    }
}
