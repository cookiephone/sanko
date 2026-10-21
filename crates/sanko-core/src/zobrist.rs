//! Zobrist hashing tables.
//!
//! Tables are generated at compile time via `const fn` splitmix64 on a fixed
//! seed, so the hash values are stable across builds and platforms (important
//! for reproducible bug reports).
//!
//! The table is per-`(point, color)`. The Board owns position-only hash.
//! Side-to-move XOR is left to the search/TT layer. A second, independent
//! table gives each board a 128-bit key (see `Board::key`), exact on boards of
//! up to 49 points.

use crate::color::Color;

const SEED_BLACK: u64 = 0xC0FF_EEC0_DEC0_DEC0;
const SEED_WHITE: u64 = 0xDEAD_BEEF_CAFE_F00D;
/// Seeds of the second table (digits of pi, far from the first seeds).
const SEED2_BLACK: u64 = 0x243F_6A88_85A3_08D3;
const SEED2_WHITE: u64 = 0x1319_8A2E_0370_7344;

/// David Stafford's variant of the splitmix64 finalizer (Mix13).
#[inline]
const fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[derive(Clone)]
pub(crate) struct Zobrist<const NPOINTS: usize> {
    by_point_color: [[u64; 2]; NPOINTS],
}

impl<const NPOINTS: usize> Zobrist<NPOINTS> {
    pub(crate) const fn new() -> Self {
        Self::seeded(SEED_BLACK, SEED_WHITE)
    }

    /// The second table, independent of the first.
    pub(crate) const fn second() -> Self {
        Self::seeded(SEED2_BLACK, SEED2_WHITE)
    }

    const fn seeded(black: u64, white: u64) -> Self {
        let mut t = [[0u64; 2]; NPOINTS];
        let mut i = 0;
        while i < NPOINTS {
            t[i][0] = splitmix64(black.wrapping_add(i as u64));
            t[i][1] = splitmix64(white.wrapping_add(i as u64));
            i += 1;
        }
        Self { by_point_color: t }
    }

    #[inline]
    pub(crate) const fn key(&self, point: usize, color: Color) -> u64 {
        self.by_point_color[point][color.index()]
    }
}

/// Hash key that distinguishes side-to-move. XOR'd by the search layer when
/// building TT keys (PSK history keys are position-only, with no STM).
pub const STM_BLACK: u64 = 0xA5A5_5A5A_FACE_BEEF;
pub const STM_WHITE: u64 = 0x5A5A_A5A5_DEAD_F00D;

/// XOR key for "consecutive passes" counter in TT keys. Not part of board hash.
pub const PASS_KEY: u64 = 0x1234_5678_9ABC_DEF0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_distinct() {
        const NPOINTS: usize = 25;
        let z: Zobrist<NPOINTS> = Zobrist::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..NPOINTS {
            assert!(
                seen.insert(z.key(i, Color::Black)),
                "collision at {i} Black"
            );
            assert!(
                seen.insert(z.key(i, Color::White)),
                "collision at {i} White"
            );
        }
        assert!(seen.iter().all(|&k| k != 0), "no key should be zero");
    }

    /// Board hashes are XORs of keys, so distinct boards have distinct hashes
    /// exactly when the 2N keys are linearly independent over GF(2). The search's
    /// position key also XORs in one side-to-move key and possibly `PASS_KEY`,
    /// so it is exact when the 2N keys, `STM_BLACK ^ STM_WHITE` and `PASS_KEY`
    /// are independent. Keys depend only on the point index, so the largest
    /// board (30 points) covers all. At 31 points the rank falls short.
    #[test]
    fn keys_are_linearly_independent() {
        let z: Zobrist<30> = Zobrist::new();
        let mut basis = [0u64; 64];
        let mut rank = 0;
        let mut gens: Vec<u64> = (0..30).flat_map(|i| [z.key(i, Color::Black), z.key(i, Color::White)]).collect();
        gens.extend([STM_BLACK ^ STM_WHITE, PASS_KEY]);
        for mut v in gens {
            for bit in (0..64).rev() {
                if (v >> bit) & 1 == 0 {
                    continue;
                }
                if basis[bit] == 0 {
                    basis[bit] = v;
                    rank += 1;
                    break;
                }
                v ^= basis[bit];
            }
        }
        assert_eq!(rank, 62);
    }

    /// The 128-bit board key (`Board::key`) XORs, for each stone, the pair of its
    /// keys in the two tables. It is exact when the 2N pairs are linearly
    /// independent as 128-bit vectors over GF(2). Keys depend only on the point
    /// index, so 49 points (7×7, the largest supported board) cover all boards.
    #[test]
    fn board_keys_are_linearly_independent_to_49_points() {
        let (z, z2): (Zobrist<49>, Zobrist<49>) = (Zobrist::new(), Zobrist::second());
        let gens: Vec<u128> = (0..49)
            .flat_map(|i| {
                [Color::Black, Color::White].map(|c| u128::from(z.key(i, c)) | (u128::from(z2.key(i, c)) << 64))
            })
            .collect();
        let mut basis = [0u128; 128];
        let mut rank = 0;
        for mut v in gens {
            for bit in (0..128).rev() {
                if (v >> bit) & 1 == 0 {
                    continue;
                }
                if basis[bit] == 0 {
                    basis[bit] = v;
                    rank += 1;
                    break;
                }
                v ^= basis[bit];
            }
        }
        assert_eq!(rank, 98);
    }

    #[test]
    fn keys_are_deterministic() {
        let z: Zobrist<4> = Zobrist::new();
        let snapshot = [
            z.key(0, Color::Black),
            z.key(0, Color::White),
            z.key(1, Color::Black),
            z.key(3, Color::White),
        ];
        let expected = [
            splitmix64(SEED_BLACK),
            splitmix64(SEED_WHITE),
            splitmix64(SEED_BLACK.wrapping_add(1)),
            splitmix64(SEED_WHITE.wrapping_add(3)),
        ];
        assert_eq!(snapshot, expected);
    }
}
