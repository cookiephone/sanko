//! Transposition table for the df-pn solver ([`crate::dfpn`]).
//!
//! df-pn keeps its whole search state here: unproven entries carry the
//! proof/disproof numbers (`phi`, `delta`, from the side to move's view) that
//! the threshold descent re-reads on every revisit, and proven entries carry
//! the winner plus the **GHI proof attributes** that decide whether the proof
//! transfers to a different path (see `dfpn.rs` "GHI under positional
//! superko"):
//!
//! - `bloom`: first word of a 128-bit Bloom filter of the *canonical* hashes of
//!   the boards the proof's winner produces (the second word is kept in `sig`,
//!   see below). A path whose ancestors miss the filter cannot make any winner
//!   move of the proof superko-illegal.
//! - superko exclusion targets: proofs whose loser moves were excluded by
//!   repeating an ancestor board are path-specific. The solver stores them
//!   under a twin key with the `many` flag (they are ~2% of proofs), so the
//!   table needs no target field.
//! - `sig`: for a path-specific (`many`) entry, the set-hash of the path's boards
//!   when the proof was made: the entry is reused only on a path with the same
//!   board set and raw position, the *identical* superko game. A transferable
//!   entry is always reused through the full transfer test and needs no path,
//!   so its `sig` holds the second word of the Bloom filter.
//! - `min_stones`: fewest stones on any winner-produced board of the proof.
//! - `cb`/`cw`: the black/white stones present on *every* winner-produced
//!   board (canonical frame).
//! - `terr`: union of the winner's static territory over the proof's static
//!   leaves (canonical frame). A path on which a winner stone was ever captured
//!   inside it could make the static mop-up superko-illegal.
//!
//! ## Layout
//!
//! Two tables, so the bulk of the state (unproven numbers) packs densely:
//!
//! - **unproven**: one-word (8-byte) entries, eight per 64-byte bucket:
//!   `phi`/`delta` as 15-bit floats (exact below 2¹⁰, < 0.2% error above, and
//!   they only steer the search), a log work estimate, the base-entry `twinned`
//!   flag and a 26-bit key check. Proofs live in the other table with a
//!   128-bit identity.
//! - **proofs**: 48-byte entries, four per 192-byte bucket: Bloom, path
//!   signature, second hash, and two packed words (one with the 25-point
//!   common-stone sets, min stones, symmetry and flags, the other with the
//!   32-point territory, PV and work). Boards of 26 to 32 points use the **mid**
//!   layout: the same 48-byte entries, one word with the two 32-point
//!   common-stone sets and one with territory, PV, work, min stones, symmetry
//!   and flags. Larger boards use the **wide** layout: 64-byte entries, four per
//!   256-byte bucket, with full 64-bit territory and common-stone words.
//!
//! Entries are **self-validating**: `xkey` stores the XOR of the key and the
//! other words, so the words a reader loaded must reproduce the probe key, and
//! the second hash `check` is stored sealed with a non-linear hash of the other
//! words. A slot mixing two writes (a torn read, or two writers interleaving)
//! fails the seal except with probability about 2^-64, so it is a miss rather
//! than a corrupt entry. No locks, and no `unsafe` apart from the cache
//! prefetch hint in `prefetch`. `Relaxed` atomics compile to plain
//! loads/stores on x86.
//!
//! ## False matches and eviction
//!
//! Both cost only time. A false match on an unproven entry's 26 check bits
//! gives wrong numbers, but never zero ones (an unproven entry is stored only
//! with both numbers at least 1), so it steers the search without proving
//! anything. A false `twinned` flag costs one extra twin probe, which is fully
//! checked. An evicted proof, or an evicted `twinned` flag that hides a twin
//! proof, means the position is searched again.

use std::sync::atomic::{AtomicU64, Ordering};

/// Ask the CPU to bring the cache line holding `x` into the cache without
/// waiting for it. Unlike a load, a prefetch never stalls the core while the
/// line is on its way, so several table misses overlap with useful work.
#[inline]
#[allow(unsafe_code)]
pub(crate) fn prefetch<T>(x: &T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `_mm_prefetch` is only a cache hint: it never faults, not even
    // for an invalid address, and has no architectural effect. Its SSE
    // requirement is part of the x86-64 baseline. `x` is a valid reference.
    unsafe {
        core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(std::ptr::from_ref(x).cast::<i8>());
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = x;
}

/// A vector of `n` values made by `make`, written by `threads` threads at once
/// (the solver's own thread count). The tables run to hundreds of GiB, and
/// writing them from one thread took most of a minute (it also places the first
/// touch of every page).
#[allow(unsafe_code)]
pub(crate) fn filled_vec<T: Send>(n: usize, threads: usize, make: impl Fn() -> T + Sync) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    let chunk = n.div_ceil(threads.max(1)).max(1 << 16);
    std::thread::scope(|s| {
        for part in v.spare_capacity_mut()[..n].chunks_mut(chunk) {
            let make = &make;
            s.spawn(move || {
                for x in part {
                    x.write(make());
                }
            });
        }
    });
    // SAFETY: the scope above joined every thread, and the threads wrote all of
    // the first `n` elements (the chunks cover `[..n]` exactly).
    unsafe { v.set_len(n) };
    v
}

/// Solved proof/disproof number (as stored in [`PnEntry::phi`]/`delta`).
pub const INF: u64 = u64::MAX;

/// Proof status of an entry, from the side to move's point of view.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PnStatus {
    Unproven,
    /// The side to move wins.
    Win,
    /// The side to move loses.
    Loss,
}

/// One decoded table entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PnEntry {
    pub key: u64,
    pub status: PnStatus,
    /// Unproven: proof number of the side to move (approximate, see module docs).
    pub phi: u64,
    /// Unproven: disproof number of the side to move.
    pub delta: u64,
    /// Proven: Bloom filter of canonical winner-produced board hashes.
    pub bloom: u64,
    /// Proven: the proof depends on a superko exclusion target, a symmetry
    /// merge or (narrow layout) territory beyond 32 points, so it is valid only
    /// on its exact path. Must survive the TT
    /// round trip: a parent composed from it inherits the dependency.
    pub many: bool,
    /// Proven: path-specific entries, the set-signature of the path the proof
    /// was made on; transferable entries, the second Bloom word.
    pub sig: u64,
    /// Symmetry index that canonicalized the stored node's raw board. Equal
    /// symmetry + equal canonical key ⇔ the identical raw board.
    pub sym: u8,
    /// Proven win: winning move (`Move::encode`) in the canonical frame.
    pub pv: u8,
    /// Proven: fewest stones on any winner-produced board of the proof.
    pub min_stones: u8,
    /// Proven: black / white stones common to every winner-produced board,
    /// canonical frame (`0` = no information).
    pub cb: u64,
    pub cw: u64,
    /// Proven: union of the winner's static territory over the proof's static
    /// leaves, canonical frame.
    pub terr: u64,
    /// Proven: an independent second 64-bit hash of the canonical position.
    /// The base key alone is exact on boards of up to 30 points: its 60 point
    /// keys and the side-to-move and pass keys are linearly independent over
    /// GF(2) (tested in `sanko-core`'s `zobrist.rs`), and the alternate-key
    /// constants lie outside their span (tested in `dfpn.rs`), so base and
    /// alternate keys never collide. This check guards twin keys, which add a
    /// non-linear path signature and can collide, and slots mixed by concurrent
    /// writes (see the seal above). A probe compares up to
    /// four keys that already share the bucket's index bits (31 of them at
    /// 2^31 buckets), so two unrelated keys agree with probability about 2^-33
    /// per slot. Over the ~6·10¹² probes of the largest runs, false matches
    /// involving twin keys are expected in the tens to hundreds.
    pub check: u64,
    /// Base entries: a path-specific twin entry exists for this position.
    pub twinned: bool,
    /// Search effort spent below this node (replacement priority).
    pub work: u64,
}

impl PnEntry {
    #[must_use]
    pub const fn unproven(key: u64, phi: u64, delta: u64, work: u64) -> Self {
        Self {
            key,
            status: PnStatus::Unproven,
            phi,
            delta,
            bloom: 0,
            many: false,
            sig: 0,
            sym: 0,
            pv: 0,
            min_stones: 0,
            cb: 0,
            cw: 0,
            terr: 0,
            check: 0,
            twinned: false,
            work,
        }
    }
}

// ---------------------------------------------------------------------------
// Unproven table encoding.
// ---------------------------------------------------------------------------

const OCCUPIED: u64 = 1 << 63;
const U_TWINNED: u64 = 1 << 62;
const U_WORK_SHIFT: u64 = 56; // 6 bits
const U_DELTA_SHIFT: u64 = 41; // 15 bits
const U_PHI_SHIFT: u64 = 26; // 15 bits
const F15_MASK: u64 = (1 << 15) - 1;
/// Key check bits stored in an unproven word (bits 38..64 of the key).
const U_CHECK_MASK: u64 = (1 << 26) - 1;

#[inline]
const fn ucheck(key: u64) -> u64 {
    (key >> 38) & U_CHECK_MASK
}

/// 15-bit float: 5-bit exponent, 10-bit mantissa (exact below 2^10, then
/// rounding down with < 0.2% relative error, saturating near 2.2e12).
/// Monotone.
#[inline]
const fn enc15(v: u64) -> u64 {
    if v < (1 << 10) {
        v
    } else {
        let e = (64 - v.leading_zeros() as u64) - 10;
        if e > 31 { (31 << 10) | 1023 } else { (e << 10) | (v >> e) }
    }
}

#[inline]
const fn dec15(c: u64) -> u64 {
    let e = c >> 10;
    let m = c & ((1 << 10) - 1);
    if e == 0 { m } else { m << e }
}

/// Bit length of the work count, 6 bits (replacement ranking only).
#[inline]
const fn enc_work6(w: u64) -> u64 {
    64 - w.leading_zeros() as u64
}

// ---------------------------------------------------------------------------
// Proof table encoding. Narrow: words [bloom, sig, check, a, b] (48-byte entries):
//   a = black-common:25 | white-common:25 | min_stones:6 | sym:3 | loss | many | . | . | occupied
//   b = terr:32 | pv:7 | work:25
// Mid: words [bloom, sig, check, a, b] (48-byte entries, boards of 26 to 32 points):
//   a = terr:32 | pv:7 | work:11 (float) | min_stones:6 | sym:3 | loss | many | . | . | occupied
//   b = black-common:32 | white-common:32
// Wide: words [bloom, sig, check, a, terr, cb, cw] (64-byte entries):
//   a = pv:7 | work:25 | . | min_stones:6 | sym:3 | loss | many | . | . | occupied
// ---------------------------------------------------------------------------

const PA_WHITE_SHIFT: u64 = 25;
const PA_MINS_SHIFT: u64 = 50;
const PA_SYM_SHIFT: u64 = 56;
const PA_LOSS: u64 = 1 << 59;
const PA_MANY: u64 = 1 << 60;
const C25: u64 = (1 << 25) - 1;
const PB_PV_SHIFT: u64 = 32;
const PB_WORK_SHIFT: u64 = 39;
const PB_WORK_MAX: u64 = (1 << 25) - 1;
const PW_WORK_SHIFT: u64 = 7;
const PM_PV_SHIFT: u64 = 32;
const PM_WORK_SHIFT: u64 = 39;
const C32: u64 = 0xFFFF_FFFF;

/// 11-bit float of a work count: 5-bit exponent, 6-bit mantissa (exact below
/// 64, then rounded down). Monotone, so codes compare like the counts.
#[inline]
const fn enc11(v: u64) -> u64 {
    if v < (1 << 6) {
        v
    } else {
        let e = (64 - v.leading_zeros() as u64) - 6;
        if e > 31 { (31 << 6) | 63 } else { (e << 6) | (v >> e) }
    }
}

#[inline]
const fn dec11(c: u64) -> u64 {
    let e = c >> 6;
    let m = c & 63;
    if e == 0 { m } else { m << e }
}

/// Proof-entry layout, chosen by board size (see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// 48-byte entries, common-stone sets of up to 25 points.
    Narrow,
    /// 48-byte entries for boards of 26 to 32 points.
    Mid,
    /// 64-byte entries with full 64-bit territory and common-stone sets.
    Wide,
}

impl Layout {
    /// The most compact layout that keeps full information for `npoints` points.
    #[must_use]
    pub const fn for_points(npoints: usize) -> Self {
        if npoints > 32 {
            Self::Wide
        } else if npoints > 25 {
            Self::Mid
        } else {
            Self::Narrow
        }
    }
}

/// One self-validating proof entry of `W` data words.
#[derive(Debug)]
struct Slot<const W: usize> {
    xkey: AtomicU64,
    w: [AtomicU64; W],
}

impl<const W: usize> Default for Slot<W> {
    fn default() -> Self {
        Self { xkey: AtomicU64::new(0), w: std::array::from_fn(|_| AtomicU64::new(0)) }
    }
}

/// Eight one-word unproven entries: `[occupied | twinned | work:6 | delta:15 |
/// phi:15 | key check:26]`. A single atomic word cannot tear. Only the 26
/// check bits (plus the bucket index) identify the key: a rare false match
/// merely misguides the search, since unproven numbers are heuristics, never proof.
#[derive(Debug, Default)]
#[repr(C, align(64))]
struct UBucket {
    slots: [AtomicU64; 8],
}

#[derive(Debug)]
#[repr(C, align(64))]
struct PBucket<const W: usize> {
    slots: [Slot<W>; 4],
}

impl<const W: usize> Default for PBucket<W> {
    fn default() -> Self {
        Self { slots: std::array::from_fn(|_| Slot::default()) }
    }
}

/// Words of a narrow (48-byte) and a wide (64-byte) proof entry.
const NW: usize = 5;
const WW: usize = 7;

/// The df-pn transposition table (see module docs).
#[derive(Debug)]
pub struct PnTable {
    unproven: Vec<UBucket>,
    /// Exactly one of the two proof tables is allocated (see `wide`).
    proofs: Vec<PBucket<NW>>,
    wproofs: Vec<PBucket<WW>>,
    wide: bool,
    /// The narrow-size table uses the mid encoding (boards of 26 to 32 points).
    mid: bool,
    umask: u64,
    /// Proof buckets, and `np - 1` when `np` is a power of two (else 0): such a
    /// table is indexed by the mixed key's low bits, any other size by the
    /// mixed key scaled to `np` (multiply-shift).
    np: u64,
    pmask: u64,
}

/// Sampled occupancy of a [`PnTable`] (see [`PnTable::fill`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TableFill {
    /// Fraction of the unproven-entry slots in use.
    pub unproven: f64,
    /// Fraction of the proof slots in use.
    pub proofs: f64,
    /// Of the proof slots in use, the fraction holding path-specific proofs.
    pub path_specific: f64,
}

/// Up to `samples` indices spread evenly over `0..len`.
fn sample_indices(len: usize, samples: usize) -> impl Iterator<Item = usize> {
    let n = samples.min(len);
    let step = len.checked_div(n).unwrap_or(0);
    (0..n).map(move |i| i * step)
}

fn fraction(part: u64, whole: u64) -> f64 {
    if whole == 0 { 0.0 } else { part as f64 / whole as f64 }
}

/// splitmix64 finalizer (bijective, non-linear), for [`PnTable::seal`].
#[inline]
const fn seal_mix(x: u64) -> u64 {
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
const fn mixkey(key: u64) -> u64 {
    // Canonical keys' low bits feed the static cache's index: decorrelate.
    key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 20
}

impl PnTable {
    /// `2^log_buckets` unproven buckets (8 × 8 B) and `2^proof_log` proof
    /// buckets (4 × 48 B, or 4 × 64 B when `wide`), in the narrow or the wide
    /// layout, filled by one thread (small tables). See [`Self::with_layout`].
    #[must_use]
    pub fn new(log_buckets: u32, proof_log: u32, wide: bool) -> Self {
        Self::with_layout(log_buckets, proof_log, if wide { Layout::Wide } else { Layout::Narrow }, 1)
    }

    /// `2^log_buckets` unproven buckets (8 × 8 B) and `2^proof_log` proof
    /// buckets in `layout` (4 × 48 B for narrow and mid, 4 × 64 B for wide),
    /// filled by `threads` threads.
    #[must_use]
    pub fn with_layout(log_buckets: u32, proof_log: u32, layout: Layout, threads: usize) -> Self {
        assert!(proof_log <= 34, "bad table size");
        Self::with_sizes(log_buckets, 1usize << proof_log, layout, threads)
    }

    /// Like [`Self::with_layout`], with any number of proof buckets (from 1 to
    /// 2^34), so that the proof table can fill a memory budget. The bucket
    /// index only spreads the keys: entries are identified by their full key
    /// and second hash whatever the index.
    #[must_use]
    pub fn with_sizes(log_buckets: u32, proof_buckets: usize, layout: Layout, threads: usize) -> Self {
        let wide = layout == Layout::Wide;
        assert!(log_buckets <= 34 && (1..=1 << 34).contains(&proof_buckets), "bad table size");
        let nu = 1usize << log_buckets;
        let np = proof_buckets;
        let unproven = filled_vec(nu, threads, UBucket::default);
        let (proofs, wproofs) = if wide {
            (Vec::new(), filled_vec(np, threads, PBucket::default))
        } else {
            (filled_vec(np, threads, PBucket::default), Vec::new())
        };
        Self {
            unproven,
            proofs,
            wproofs,
            wide,
            mid: layout == Layout::Mid,
            umask: (nu as u64) - 1,
            np: np as u64,
            pmask: if np.is_power_of_two() { (np as u64) - 1 } else { 0 },
        }
    }

    /// Bytes of table storage.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.unproven.len() * std::mem::size_of::<UBucket>()
            + self.proofs.len() * std::mem::size_of::<PBucket<NW>>()
            + self.wproofs.len() * std::mem::size_of::<PBucket<WW>>()
    }

    /// Occupancy estimated from up to `samples` buckets of each part, spread
    /// evenly over it (bucket indices are hashed, so the estimate is unbiased).
    /// Read-only, for progress displays: concurrent writers can make it a
    /// little stale, never wrong by more than the writes in flight.
    #[must_use]
    pub fn fill(&self, samples: usize) -> TableFill {
        let (mut used, mut total) = (0, 0);
        for i in sample_indices(self.unproven.len(), samples) {
            for s in &self.unproven[i].slots {
                total += 1;
                used += u64::from(s.load(Ordering::Relaxed) & OCCUPIED != 0);
            }
        }
        // Word 3 holds the flags in both proof layouts.
        let flags: Vec<u64> = if self.wide {
            sample_indices(self.wproofs.len(), samples)
                .flat_map(|i| self.wproofs[i].slots.iter().map(|s| s.w[3].load(Ordering::Relaxed)))
                .collect()
        } else {
            sample_indices(self.proofs.len(), samples)
                .flat_map(|i| self.proofs[i].slots.iter().map(|s| s.w[3].load(Ordering::Relaxed)))
                .collect()
        };
        let in_use = flags.iter().filter(|&&a| a & OCCUPIED != 0).count() as u64;
        let many = flags.iter().filter(|&&a| a & OCCUPIED != 0 && a & PA_MANY != 0).count() as u64;
        TableFill {
            unproven: fraction(used, total),
            proofs: fraction(in_use, flags.len() as u64),
            path_specific: fraction(many, in_use),
        }
    }

    /// Whether proofs keep full 64-bit territory (else only 32 bits: callers
    /// must store proofs with wider territory as path-specific).
    #[must_use]
    pub const fn wide(&self) -> bool {
        self.wide
    }

    #[inline]
    fn ubucket(&self, key: u64) -> &UBucket {
        &self.unproven[(mixkey(key) & self.umask) as usize]
    }

    /// The proof bucket index of `key` (see `np`). `mixkey` has 44 bits.
    #[inline]
    const fn pindex(&self, key: u64) -> usize {
        let m = mixkey(key);
        if self.pmask != 0 || self.np == 1 {
            (m & self.pmask) as usize
        } else {
            ((m as u128 * self.np as u128) >> 44) as usize
        }
    }

    #[inline]
    fn pbucket(&self, key: u64) -> &PBucket<NW> {
        &self.proofs[self.pindex(key)]
    }

    #[inline]
    fn wbucket(&self, key: u64) -> &PBucket<WW> {
        &self.wproofs[self.pindex(key)]
    }

    #[inline]
    fn load<const W: usize>(s: &Slot<W>) -> (u64, [u64; W]) {
        let mut w = [0u64; W];
        for (d, a) in w.iter_mut().zip(&s.w) {
            *d = a.load(Ordering::Relaxed);
        }
        (s.xkey.load(Ordering::Relaxed), w)
    }

    /// Nonlinear seal over a proof entry's words other than `check` (word 2).
    /// The stored check word is `check ^ seal`. The key fold below is linear, so
    /// a slot assembled from two different writes (a reader overtaking a writer,
    /// or two writers interleaving on one slot) can still fold to a valid key.
    /// Its seal then differs from the one the check word was sealed with, so the
    /// decoded second hash is random and the 128-bit check rejects the slot
    /// (probability 1 − 2⁻⁶⁴).
    #[inline]
    fn seal<const W: usize>(w: &[u64; W]) -> u64 {
        let mut h = 0u64;
        for (i, &x) in w.iter().enumerate() {
            if i != 2 {
                h ^= seal_mix(x ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            }
        }
        h
    }

    /// The key a loaded slot holds (word 3 carries the flags in both layouts).
    #[inline]
    fn slot_key<const W: usize>(xkey: u64, w: &[u64; W]) -> Option<u64> {
        if w[3] & OCCUPIED == 0 {
            return None;
        }
        Some(w.iter().fold(xkey, |k, &x| k ^ x))
    }

    fn write<const W: usize>(s: &Slot<W>, key: u64, w: [u64; W]) {
        s.xkey.store(w.iter().fold(key, |k, &x| k ^ x), Ordering::Relaxed);
        for (a, v) in s.w.iter().zip(w) {
            a.store(v, Ordering::Relaxed);
        }
    }

    /// Decode the flag word `a` shared by both layouts.
    const fn decode_common(key: u64, bloom: u64, sig: u64, check: u64, a: u64) -> PnEntry {
        let loss = a & PA_LOSS != 0;
        PnEntry {
            key,
            status: if loss { PnStatus::Loss } else { PnStatus::Win },
            phi: if loss { INF } else { 0 },
            delta: if loss { 0 } else { INF },
            bloom,
            many: a & PA_MANY != 0,
            sig,
            cb: 0,
            cw: 0,
            terr: 0,
            check,
            sym: ((a >> PA_SYM_SHIFT) & 7) as u8,
            pv: 0,
            min_stones: ((a >> PA_MINS_SHIFT) & 63) as u8,
            twinned: false,
            work: 0,
        }
    }

    fn probe_proof(&self, key: u64) -> Option<PnEntry> {
        if self.wide {
            for slot in &self.wbucket(key).slots {
                let (xkey, w) = Self::load(slot);
                if Self::slot_key(xkey, &w) == Some(key) {
                    let [bloom, sig, check, a, terr, cb, cw] = w;
                    let mut e = Self::decode_common(key, bloom, sig, check ^ Self::seal(&w), a);
                    e.cb = cb;
                    e.cw = cw;
                    e.terr = terr;
                    e.pv = (a & 0x7F) as u8;
                    e.work = (a >> PW_WORK_SHIFT) & PB_WORK_MAX;
                    return Some(e);
                }
            }
            return None;
        }
        for slot in &self.pbucket(key).slots {
            let (xkey, w) = Self::load(slot);
            if Self::slot_key(xkey, &w) == Some(key) {
                let [bloom, sig, check, a, b] = w;
                let mut e = Self::decode_common(key, bloom, sig, check ^ Self::seal(&w), a);
                if self.mid {
                    e.terr = a & C32;
                    e.pv = ((a >> PM_PV_SHIFT) & 0x7F) as u8;
                    e.work = dec11((a >> PM_WORK_SHIFT) & 0x7FF);
                    e.cb = b & C32;
                    e.cw = b >> 32;
                    return Some(e);
                }
                e.cb = a & C25;
                e.cw = (a >> PA_WHITE_SHIFT) & C25;
                e.terr = b & 0xFFFF_FFFF;
                e.pv = ((b >> PB_PV_SHIFT) & 0x7F) as u8;
                e.work = (b >> PB_WORK_SHIFT) & PB_WORK_MAX;
                return Some(e);
            }
        }
        None
    }

    fn probe_unproven(&self, key: u64) -> Option<PnEntry> {
        let c = ucheck(key);
        for slot in &self.ubucket(key).slots {
            let d = slot.load(Ordering::Relaxed);
            if d & OCCUPIED != 0 && d & U_CHECK_MASK == c {
                let mut e = PnEntry::unproven(
                    key,
                    dec15((d >> U_PHI_SHIFT) & F15_MASK),
                    dec15((d >> U_DELTA_SHIFT) & F15_MASK),
                    1u64 << ((d >> U_WORK_SHIFT) & 63),
                );
                e.twinned = d & U_TWINNED != 0;
                return Some(e);
            }
        }
        None
    }

    /// Prefetch the lines a later [`Self::probe`] of `key` will read (the
    /// unproven bucket and every line of the proof bucket), so several probes'
    /// cache misses overlap without stalling the core.
    #[inline]
    pub fn touch(&self, key: u64) {
        // The first words of the four slots cover every line of the bucket
        // (64-byte slots in the wide layout, 48-byte slots in the others).
        if self.wide {
            for slot in &self.wbucket(key).slots {
                prefetch(&slot.w[0]);
            }
        } else {
            for slot in &self.pbucket(key).slots {
                prefetch(&slot.w[0]);
            }
        }
        prefetch(&self.ubucket(key).slots[0]);
    }

    /// Prefetch the unproven bucket of `key` only (see [`Self::touch`]).
    #[inline]
    pub fn touch_unproven(&self, key: u64) {
        prefetch(&self.ubucket(key).slots[0]);
    }

    /// The numbers `(phi, delta)` of the unproven entry for `key`, without
    /// consulting the proof table (one cache line instead of five). For reading
    /// other workers' progress, which only steers the search.
    #[must_use]
    pub fn probe_numbers(&self, key: u64) -> Option<(u64, u64)> {
        self.probe_unproven(key).map(|e| (e.phi, e.delta))
    }

    /// The proof for `key` if one is stored, else its unproven entry.
    #[must_use]
    pub fn probe(&self, key: u64) -> Option<PnEntry> {
        self.probe_proof(key).or_else(|| self.probe_unproven(key))
    }

    /// Store an entry: unproven ones go to the dense table (keeping a base
    /// entry's `twinned` flag), proofs to the proof table (same key overwritten,
    /// else an empty slot, else the one with least work).
    pub fn store(&self, e: &PnEntry) {
        if e.status == PnStatus::Unproven {
            self.store_unproven(e.key, e.phi, e.delta, e.work, e.twinned);
        } else {
            self.store_proof(e);
        }
    }

    fn store_unproven(&self, key: u64, phi: u64, delta: u64, work: u64, twinned: bool) {
        let c = ucheck(key);
        let mut data = OCCUPIED
            | c
            | (enc15(phi) << U_PHI_SHIFT)
            | (enc15(delta) << U_DELTA_SHIFT)
            | (enc_work6(work) << U_WORK_SHIFT);
        if twinned {
            data |= U_TWINNED;
        }
        let bucket = self.ubucket(key);
        let mut victim = 0usize;
        let mut victim_pri = u64::MAX;
        for (i, slot) in bucket.slots.iter().enumerate() {
            let d = slot.load(Ordering::Relaxed);
            if d & OCCUPIED == 0 {
                if victim_pri != 0 {
                    victim = i;
                    victim_pri = 0;
                }
                continue;
            }
            if d & U_CHECK_MASK == c {
                slot.store(data | (d & U_TWINNED), Ordering::Relaxed);
                return;
            }
            let pri = (d >> U_WORK_SHIFT) & 63;
            if pri < victim_pri {
                victim_pri = pri;
                victim = i;
            }
        }
        bucket.slots[victim].store(data, Ordering::Relaxed);
    }

    /// Store a proof. The entry has no room for a superko exclusion target,
    /// nor (narrow layout) for territory beyond 32 points, so callers must
    /// store such proofs as path-specific (`many`). In the narrow layout,
    /// common-stone sets wider than 25 points are dropped (no filter, which
    /// is conservative).
    fn store_proof(&self, e: &PnEntry) {
        debug_assert!(self.wide || e.terr >> 32 == 0 || e.many, "territory must fit in 32 bits");
        let mut a = OCCUPIED
            | ((u64::from(e.min_stones).min(63)) << PA_MINS_SHIFT)
            | ((u64::from(e.sym) & 7) << PA_SYM_SHIFT);
        if e.status == PnStatus::Loss {
            a |= PA_LOSS;
        }
        if e.many {
            a |= PA_MANY;
        }
        if self.wide {
            a |= (u64::from(e.pv) & 0x7F) | (e.work.min(PB_WORK_MAX) << PW_WORK_SHIFT);
            let mut words = [e.bloom, e.sig, 0, a, e.terr, e.cb, e.cw];
            words[2] = e.check ^ Self::seal(&words);
            let slots = &self.wbucket(e.key).slots;
            let v = Self::victim(e.key, slots, |w| (w[3] >> PW_WORK_SHIFT) & PB_WORK_MAX);
            Self::write(&slots[v], e.key, words);
        } else if self.mid {
            debug_assert!(e.cb <= C32 && e.cw <= C32, "common stones must fit in 32 bits");
            a |= (e.terr & C32) | ((u64::from(e.pv) & 0x7F) << PM_PV_SHIFT) | (enc11(e.work) << PM_WORK_SHIFT);
            let b = (e.cb & C32) | ((e.cw & C32) << 32);
            let mut words = [e.bloom, e.sig, 0, a, b];
            words[2] = e.check ^ Self::seal(&words);
            let slots = &self.pbucket(e.key).slots;
            let v = Self::victim(e.key, slots, |w| (w[3] >> PM_WORK_SHIFT) & 0x7FF);
            Self::write(&slots[v], e.key, words);
        } else {
            if e.cb <= C25 && e.cw <= C25 {
                a |= e.cb | (e.cw << PA_WHITE_SHIFT);
            }
            let b = (e.terr & 0xFFFF_FFFF)
                | ((u64::from(e.pv) & 0x7F) << PB_PV_SHIFT)
                | (e.work.min(PB_WORK_MAX) << PB_WORK_SHIFT);
            let mut words = [e.bloom, e.sig, 0, a, b];
            words[2] = e.check ^ Self::seal(&words);
            let slots = &self.pbucket(e.key).slots;
            let v = Self::victim(e.key, slots, |w| (w[4] >> PB_WORK_SHIFT) & PB_WORK_MAX);
            Self::write(&slots[v], e.key, words);
        }
        let key = e.key;
        // The unproven entry for this key is now dead weight: free its slot.
        let c = ucheck(key);
        for slot in &self.ubucket(key).slots {
            let d = slot.load(Ordering::Relaxed);
            if d & OCCUPIED != 0 && d & U_CHECK_MASK == c && d & U_TWINNED == 0 {
                slot.store(0, Ordering::Relaxed);
            }
        }
    }

    /// The slot to write `key` into: its own slot, else an empty one, else the
    /// one with least work (`work` reads it from the loaded words).
    fn victim<const W: usize>(key: u64, slots: &[Slot<W>; 4], work: impl Fn(&[u64; W]) -> u64) -> usize {
        let mut victim = 0usize;
        let mut victim_pri = u64::MAX;
        for (i, slot) in slots.iter().enumerate() {
            let (xkey, w) = Self::load(slot);
            match Self::slot_key(xkey, &w) {
                Some(k) if k == key => return i,
                Some(_) => {
                    let pri = work(&w);
                    if pri < victim_pri {
                        victim_pri = pri;
                        victim = i;
                    }
                }
                None => {
                    if victim_pri != 0 {
                        victim = i;
                        victim_pri = 0;
                    }
                }
            }
        }
        victim
    }

    /// Store a proven *base* entry unless a proof is already stored for the
    /// key (base proofs are write-once: a later proof for the same position
    /// came from another path and goes to an alternate slot, or in the
    /// exact-only mode to that path's twin key, so the first path's proof stays
    /// reconstructible). Returns whether it was stored.
    ///
    /// Write-once is best-effort with several workers: the probe and the store
    /// are separate, so two workers can both see no proof and the later write
    /// wins. That is harmless. Each stored base proof stands on its own, and
    /// every reader re-tests identity and transfer before using it.
    #[must_use]
    pub fn store_base_proof(&self, e: &PnEntry) -> bool {
        debug_assert_ne!(e.status, PnStatus::Unproven);
        if self.probe_proof(e.key).is_some() {
            return false;
        }
        self.store_proof(e);
        true
    }

    /// Mark the base entry for `key` as having a path-specific twin (the flag
    /// lives on the unproven entry, created if absent). Best-effort.
    pub fn mark_twinned(&self, key: u64) {
        let (phi, delta, work) = match self.probe_unproven(key) {
            Some(e) => (e.phi, e.delta, e.work),
            None => (1, 1, 0),
        };
        self.store_unproven(key, phi, delta, work, true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float15_roundtrip_and_monotone() {
        for v in [0u64, 1, 7, (1 << 10) - 1, 1 << 10, 123_456_789, 1 << 40] {
            let d = dec15(enc15(v));
            assert!(d <= v && (v - d) as f64 <= v as f64 / f64::from(1 << 9));
        }
        let mut prev = 0;
        for v in (0..10_000_000u64).step_by(9973) {
            let d = dec15(enc15(v));
            assert!(d >= prev);
            prev = d;
        }
        assert!(dec15(enc15(u64::MAX)) > 1 << 40);
    }

    #[test]
    fn roundtrip_unproven_and_proven() {
        for wide in [false, true] {
            roundtrip(wide);
        }
    }

    /// A proof table of any size stores and finds its entries, and spreads the
    /// keys over all buckets.
    #[test]
    fn proof_table_of_any_size() {
        for np in [1usize, 3, 5, 6, 7, 1000] {
            let t = PnTable::with_sizes(4, np, Layout::Mid, 1);
            let mut used = vec![false; np];
            for k in 0..(40 * np as u64) {
                let key = k.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1;
                used[t.pindex(key)] = true;
                let mut p = PnEntry::unproven(key, 0, INF, 5);
                p.status = PnStatus::Win;
                p.phi = 0;
                p.check = key.rotate_left(7);
                t.store(&p);
                assert_eq!(t.probe(key).map(|e| e.check), Some(p.check), "np={np}");
            }
            assert!(used.iter().all(|&u| u), "np={np}: some bucket never used");
        }
    }

    /// The mid layout keeps 32-point territory and common-stone sets exactly.
    #[test]
    fn mid_layout_roundtrip() {
        let t = PnTable::with_layout(4, 3, Layout::Mid, 1);
        let mut rng = sanko_core::Rng::new(9);
        for k in 0..200u64 {
            let mut p = PnEntry::unproven(1000 + k, 0, INF, rng.next_u64() >> (rng.next_u64() % 64));
            p.status = if k % 2 == 0 { PnStatus::Win } else { PnStatus::Loss };
            p.phi = if k % 2 == 0 { 0 } else { INF };
            p.delta = if k % 2 == 0 { INF } else { 0 };
            p.bloom = rng.next_u64();
            p.sig = rng.next_u64();
            p.sym = (k % 8) as u8;
            p.pv = (k % 33) as u8;
            p.min_stones = (k % 31) as u8;
            p.cb = rng.next_u64() & C32;
            p.cw = rng.next_u64() & C32;
            p.terr = rng.next_u64() & C32;
            p.check = rng.next_u64();
            p.many = k % 3 == 0;
            t.store(&p);
            let mut want = p;
            want.work = dec11(enc11(p.work));
            assert_eq!(t.probe(p.key), Some(want));
        }
        // Exact below 64, then rounded down by less than 1/32; saturating (and monotone) above ~1.4e11.
        let mut prev = 0;
        for v in [0u64, 1, 63, 64, 65, 1000, 1 << 20, 123_456_789, 1 << 36, u64::MAX] {
            let d = dec11(enc11(v));
            assert!(d <= v && d >= prev);
            if v < 1 << 36 {
                assert!((v - d) as f64 <= v as f64 / 32.0);
            }
            prev = d;
        }
    }

    fn roundtrip(wide: bool) {
        let t = PnTable::new(4, 3, wide);
        t.store(&PnEntry::unproven(42, 7, 9, 100));
        let e = t.probe(42).unwrap();
        assert_eq!((e.status, e.phi, e.delta), (PnStatus::Unproven, 7, 9));
        let mut p = PnEntry::unproven(43, 0, INF, 5);
        p.status = PnStatus::Win;
        p.bloom = 0xF0F0;
        p.sig = 0xABCD;
        p.sym = 5;
        p.pv = 12;
        p.min_stones = 3;
        p.cb = 0x10;
        p.cw = 0x3;
        p.terr = if wide { 0xF_0000_0055 } else { 0x55 };
        p.check = 0x1234_5678;
        p.many = true;
        t.store(&p);
        assert_eq!(t.probe(43), Some(p));
        assert_eq!(t.probe(44), None);
    }

    #[test]
    fn proof_shadows_unproven_and_base_is_write_once() {
        let t = PnTable::new(4, 3, false);
        t.store(&PnEntry::unproven(7, 3, 3, 9));
        let mut p = PnEntry::unproven(7, INF, 0, 5);
        p.status = PnStatus::Loss;
        assert!(t.store_base_proof(&p));
        assert_eq!(t.probe(7).map(|e| e.status), Some(PnStatus::Loss));
        let mut q = p;
        q.status = PnStatus::Win;
        q.phi = 0;
        q.delta = INF;
        assert!(!t.store_base_proof(&q));
        assert_eq!(t.probe(7).map(|e| e.status), Some(PnStatus::Loss));
    }

    #[test]
    fn twinned_flag_survives_overwrite() {
        let t = PnTable::new(4, 3, false);
        t.mark_twinned(11);
        t.store(&PnEntry::unproven(11, 4, 5, 1));
        let e = t.probe(11).unwrap();
        assert!(e.twinned);
        assert_eq!((e.phi, e.delta), (4, 5));
    }
}
