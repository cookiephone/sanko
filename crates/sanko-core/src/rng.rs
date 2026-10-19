//! Small deterministic PRNG for the randomized self-tests and benchmark suites.

/// Tiny xorshift64 PRNG. A zero seed is remapped to a fixed nonzero state.
#[derive(Copy, Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    #[inline]
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        // 0 is the absorbing state for xorshift64, so remap to a stable nonzero.
        let s = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
        Self { state: s }
    }

    #[inline]
    pub const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform integer in `0..n`. Uses Lemire's biased multiplication (the
    /// bias is negligible for the small `n` we ever pass, on the order of NPOINTS).
    #[inline]
    pub fn next_below(&mut self, n: u32) -> u32 {
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u32
    }
}
