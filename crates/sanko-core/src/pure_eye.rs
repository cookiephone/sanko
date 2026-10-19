//! Pure-eye analysis: our variant of Benson's algorithm (Benson 1976, "Life in
//! the Game of Go") for **unconditionally alive** chains and their territory.
//! It counts only *pure* eyes, regions all of whose points (not just the empty
//! ones) touch the chain, which keeps the territory claim sound when the owner
//! must capture intruders under positional superko (see `analyze_color` and the
//! paper's static-leaf lemma).
//!
//! A chain is unconditionally alive iff it survives no matter how the opponent
//! plays: the opponent cannot capture it given infinite tenuki by its own
//! color. Operationally this is the strongest static-life certificate Go has.
//!
//! ## Algorithm sketch
//!
//! For a fixed color `X` (complement `Y`):
//!
//! 1. Enumerate the `X`-chains.
//! 2. Enumerate the `X`-regions: maximal 4-connected sets of non-`X` points
//!    (empties and `Y`-stones together).
//! 3. For each region `R` and chain `C`, mark `R` as **vital** to `C` iff every
//!    point of `R` (empty or `Y`-stone) is orthogonally adjacent to `C`
//!    (stricter than Benson's empty-points test, see `analyze_color`).
//! 4. Let `S` start as the set of all `X`-chains. Iteratively remove any chain
//!    that does not have ≥ 2 **healthy** regions (vital to `C` AND every
//!    bordering chain still in `S`). Repeat until stable.
//! 5. Fixed-point `S` is the set of unconditionally alive `X`-chains.
//!
//! ## Cost
//!
//! For a board with `N` points, `k` chains, `r` regions, work is roughly
//! `O((k·r + N) · iterations)`, iterations bounded by `k ≤ N/2`. Microseconds
//! on 5×5/7×7.

use crate::bitboard::BitBoard;
use crate::board::Board;
use crate::color::Color;
use crate::topology::dilate;

/// Result of the pure-eye analysis of a board.
///
/// `*_alive` are the stones in unconditionally alive chains. `*_territory` are
/// the points (empties + dead opponent stones) in regions whose every
/// bordering chain is unconditionally alive and whose every point is adjacent
/// to one of those chains: guaranteed territory under any line of play.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct PureEyeAnalysis<const LANES: usize> {
    pub black_alive: BitBoard<LANES>,
    pub white_alive: BitBoard<LANES>,
    pub black_territory: BitBoard<LANES>,
    pub white_territory: BitBoard<LANES>,
}

#[must_use]
pub fn pure_eye_analysis<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
) -> PureEyeAnalysis<LANES> {
    let (black_alive, black_territory) =
        analyze_color::<ROWS, COLS, LANES, NPOINTS>(board, Color::Black);
    let (white_alive, white_territory) =
        analyze_color::<ROWS, COLS, LANES, NPOINTS>(board, Color::White);
    PureEyeAnalysis {
        black_alive,
        white_alive,
        black_territory,
        white_territory,
    }
}

/// Cheap upper bound on `color`'s secure area (alive stones + territory):
/// its stones plus every region (non-`color` component) all of whose points
/// touch its stones (alive chains are stones, and territory regions must touch
/// alive chains everywhere).
#[must_use]
pub fn secure_upper_bound<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
    color: Color,
) -> u32 {
    let stones = board.stones(color);
    if stones.is_empty() {
        return 0;
    }
    let near = dilate::<ROWS, COLS, LANES, NPOINTS>(&stones);
    let mut regions = [BitBoard::<LANES>::EMPTY; NPOINTS];
    let n = components::<ROWS, COLS, LANES, NPOINTS>(&board.empties().union(&board.stones(color.opponent())), &mut regions);
    let mut total = stones.count();
    for region in regions.iter().take(n) {
        if region.is_subset_of(&near) {
            total += region.count();
        }
    }
    total
}

/// Unconditionally alive stones and unconditional territory of one color
/// (the two colors' analyses are independent).
#[must_use]
pub fn pure_eye_analysis_of<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
    color: Color,
) -> (BitBoard<LANES>, BitBoard<LANES>) {
    analyze_color::<ROWS, COLS, LANES, NPOINTS>(board, color)
}

/// Compute unconditionally alive stones and unconditional territory for
/// `color`.
///
/// A region is *vital* (an eye) for a chain only if **every** point of it
/// (empty or opponent-occupied) is adjacent to that chain ("pure"). Benson's
/// classic test only asks this of the empty points, which is enough when the
/// owner never moves (Benson's theorem), but our static leaves let the owner
/// *mop up*: capture the opponent stones in its territory, because under area
/// scoring without dead-stone removal they must really be captured. Capturing
/// inside a classic (impure) eye frees points that need not touch the chain.
/// The opponent may re-enter there with two liberties while the chain is left
/// in atari on its other eye, and then capture it. With pure eyes every freed
/// point touches the chain again, the opponent can never fill an eye's last
/// point (no suicide, the other eye stands), and the owner only fills an eye
/// point to capture, which frees points: each alive chain keeps two liberties
/// throughout the mop-up, so it can never be captured.
fn analyze_color<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
    color: Color,
) -> (BitBoard<LANES>, BitBoard<LANES>) {
    let stones = board.stones(color);
    if stones.is_empty() {
        return (BitBoard::EMPTY, BitBoard::EMPTY);
    }
    let empties = board.empties();
    let opp_stones = board.stones(color.opponent());
    let non_color = empties.union(&opp_stones);

    let mut regions = [BitBoard::<LANES>::EMPTY; NPOINTS];
    let n_regions = components::<ROWS, COLS, LANES, NPOINTS>(&non_color, &mut regions);
    // An alive chain needs two healthy regions: with fewer than two regions
    // nothing is alive, hence no territory either.
    if n_regions < 2 {
        return (BitBoard::EMPTY, BitBoard::EMPTY);
    }
    // A region can be vital to a chain only if all its empty points touch our
    // stones, and an alive chain needs two such regions.
    let near = dilate::<ROWS, COLS, LANES, NPOINTS>(&stones);
    let mut candidates = 0;
    for region in regions.iter().take(n_regions) {
        if region.intersect(&empties).is_subset_of(&near) {
            candidates += 1;
            if candidates == 2 {
                break;
            }
        }
    }
    if candidates < 2 {
        return (BitBoard::EMPTY, BitBoard::EMPTY);
    }
    let mut chains = [BitBoard::<LANES>::EMPTY; NPOINTS];
    let n_chains = components::<ROWS, COLS, LANES, NPOINTS>(&stones, &mut chains);

    if n_chains == 0 {
        return (BitBoard::EMPTY, BitBoard::EMPTY);
    }

    let mut chain_dilations = [BitBoard::<LANES>::EMPTY; NPOINTS];
    for c in 0..n_chains {
        chain_dilations[c] = dilate::<ROWS, COLS, LANES, NPOINTS>(&chains[c]);
    }

    // Bitmask adjacency in place of nested vectors. Chain/region indices are
    // < NPOINTS, so a single BitBoard<LANES> holds the whole adjacency set.
    let mut bordering = [BitBoard::<LANES>::EMPTY; NPOINTS];
    let mut vital_regions_of = [BitBoard::<LANES>::EMPTY; NPOINTS];

    for r in 0..n_regions {
        let region = &regions[r];
        let region_dilated = dilate::<ROWS, COLS, LANES, NPOINTS>(region);
        let touching_stones = region_dilated.intersect(&stones);

        for c in 0..n_chains {
            if chains[c].intersect(&touching_stones).is_empty() {
                continue;
            }
            bordering[r].set(c);
            if region.is_subset_of(&chain_dilations[c]) {
                vital_regions_of[c].set(r);
            }
        }
    }

    let mut alive_mask = BitBoard::<LANES>::EMPTY;
    for c in 0..n_chains {
        alive_mask.set(c);
    }
    loop {
        let mut fully_alive_regions = BitBoard::<LANES>::EMPTY;
        for (r, border) in bordering.iter().enumerate().take(n_regions) {
            if !border.is_empty() && border.is_subset_of(&alive_mask) {
                fully_alive_regions.set(r);
            }
        }
        let mut changed = false;
        for (c, vital) in vital_regions_of.iter().enumerate().take(n_chains) {
            if !alive_mask.get(c) {
                continue;
            }
            let healthy = vital.intersect(&fully_alive_regions).count();
            if healthy < 2 {
                alive_mask.clear(c);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut alive_stones: BitBoard<LANES> = BitBoard::EMPTY;
    for (c, chain) in chains.iter().enumerate().take(n_chains) {
        if alive_mask.get(c) {
            alive_stones = alive_stones.union(chain);
        }
    }
    // Territory: a region whose every bordering chain is alive AND whose every
    // point (empty *or opponent-occupied*) is adjacent to one of those chains.
    //
    // Requiring only the *empty* points to be adjacent is not enough without
    // dead-stone removal: the opponent's stones must actually be captured, and
    // once a large block is captured its interior points (not adjacent to the
    // wall) are empty space in which the opponent may build a living group (van
    // der Werf's thesis, Fig. 5.8). With every point adjacent to an alive chain,
    // no point of the region can ever be an opponent eye, each filling move joins
    // an alive chain (so it is permanent and never suicide), and the owner can
    // always capture every intruder, so the region ends as the owner's area.
    let mut territory: BitBoard<LANES> = BitBoard::EMPTY;
    if !alive_stones.is_empty() {
        for r in 0..n_regions {
            let border = bordering[r];
            if border.is_empty() || !border.is_subset_of(&alive_mask) {
                continue;
            }
            let mut border_dilation: BitBoard<LANES> = BitBoard::EMPTY;
            for c in &border {
                border_dilation = border_dilation.union(&chain_dilations[c]);
            }
            if regions[r].is_subset_of(&border_dilation) {
                territory = territory.union(&regions[r]);
            }
        }
    }
    (alive_stones, territory)
}

/// 4-connected components of `seed`, written into `out[0..count]`.
fn components<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    seed: &BitBoard<LANES>,
    out: &mut [BitBoard<LANES>; NPOINTS],
) -> usize {
    let mut remaining = *seed;
    let mut count = 0usize;
    while !remaining.is_empty() {
        let first = remaining.iter().next().expect("non-empty just checked");
        let mut comp: BitBoard<LANES> = BitBoard::EMPTY;
        comp.set(first);
        loop {
            let grown = dilate::<ROWS, COLS, LANES, NPOINTS>(&comp).intersect(seed);
            if grown == comp {
                break;
            }
            comp = grown;
        }
        remaining = remaining.difference(&comp);
        out[count] = comp;
        count += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Point, define_board};

    define_board!(B3x3, 3, 3);
    define_board!(B4x4, 4, 4);
    define_board!(B5x5, 5, 5);
    define_board!(B7x7, 7, 7);

    fn pt5(r: usize, c: usize) -> Point<5, 5> {
        Point::<5, 5>::new(r, c).expect("test coord in range")
    }

    fn pt7(r: usize, c: usize) -> Point<7, 7> {
        Point::<7, 7>::new(r, c).expect("test coord in range")
    }

    #[test]
    fn empty_board_has_no_alive_stones() {
        let b = B5x5::new();
        let life = pure_eye_analysis(&b);
        assert!(life.black_alive.is_empty());
        assert!(life.white_alive.is_empty());
    }

    #[test]
    fn lone_stone_is_not_alive() {
        let mut b = B5x5::new();
        b.place(pt5(0, 0), Color::Black);
        let life = pure_eye_analysis(&b);
        assert!(life.black_alive.is_empty());
    }

    #[test]
    fn two_eye_group_in_corner_is_alive() {
        // Black wall with two eyes at (1,1) and (1,3).
        let mut b = B5x5::new();
        for c in 0..5 {
            b.place(pt5(0, c), Color::Black);
            b.place(pt5(2, c), Color::Black);
        }
        b.place(pt5(1, 0), Color::Black);
        b.place(pt5(1, 2), Color::Black);
        b.place(pt5(1, 4), Color::Black);

        let life = pure_eye_analysis(&b);
        for r in 0..3 {
            for c in 0..5 {
                if matches!(b.at(pt5(r, c)), Some(Color::Black)) {
                    assert!(life.black_alive.get(pt5(r, c).index()));
                }
            }
        }
        assert!(life.white_alive.is_empty());
    }

    #[test]
    fn single_eye_group_is_not_alive() {
        // Same wall but only one interior eye-point.
        let mut b = B5x5::new();
        for c in 0..3 {
            b.place(pt5(0, c), Color::Black);
            b.place(pt5(2, c), Color::Black);
        }
        b.place(pt5(1, 0), Color::Black);
        b.place(pt5(1, 2), Color::Black);

        let life = pure_eye_analysis(&b);
        assert!(life.black_alive.is_empty());
    }

    /// Pure versus classic eyes (the paper's Figure 2b). The top-left region
    /// passes Benson's classic test, since its empty points all touch the Black
    /// chain, so classic Benson would count two eyes and call the chain alive.
    /// Its White stones touch no Black stone, so it is no pure eye, and with
    /// only one pure eye the chain is not alive here.
    #[test]
    fn classic_vital_but_impure_region_is_no_eye() {
        let p = |r: usize, c: usize| Point::<4, 4>::new(r, c).expect("test coord in range");
        let mut b = B4x4::new();
        // Black chain (rows from the top): D4, D3, C3, C2, B2, B1, A1, D2. Its
        // other eye is C1 and D1, a pure two-point eye.
        for (r, c) in [(0, 3), (1, 3), (1, 2), (2, 2), (2, 1), (3, 1), (3, 0), (2, 3)] {
            b.place(p(r, c), Color::Black);
        }
        // White intruders in the corner: A4, B4, A3.
        for (r, c) in [(0, 0), (0, 1), (1, 0)] {
            b.place(p(r, c), Color::White);
        }
        let touches_black = |q: Point<4, 4>| q.neighbors().iter().flatten().any(|&n| b.at(n) == Some(Color::Black));
        // Classic condition: every empty point of the corner region (C4, B3, A2)
        // touches the chain.
        for (r, c) in [(0, 2), (1, 1), (2, 0)] {
            assert!(b.at(p(r, c)).is_none() && touches_black(p(r, c)));
        }
        // Pure condition fails: the White stones touch no Black stone.
        for (r, c) in [(0, 0), (0, 1), (1, 0)] {
            assert!(!touches_black(p(r, c)));
        }
        let life = pure_eye_analysis(&b);
        assert!(life.black_alive.is_empty());
        assert!(life.black_territory.is_empty());
    }

    #[test]
    fn enclosed_dead_stone_does_not_break_aliveness() {
        let mut b = B5x5::new();
        for c in 0..5 {
            b.place(pt5(0, c), Color::Black);
            b.place(pt5(2, c), Color::Black);
        }
        b.place(pt5(1, 0), Color::Black);
        b.place(pt5(1, 2), Color::Black);
        b.place(pt5(1, 4), Color::Black);
        b.place(pt5(1, 1), Color::White); // dead-on-arrival

        let life = pure_eye_analysis(&b);
        for r in 0..3 {
            for c in 0..5 {
                if matches!(b.at(pt5(r, c)), Some(Color::Black)) {
                    assert!(life.black_alive.get(pt5(r, c).index()));
                }
            }
        }
        assert!(life.white_alive.is_empty());
    }

    #[test]
    fn both_colors_alive_simultaneously() {
        let mut b = B7x7::new();
        for c in 0..5 {
            b.place(pt7(0, c), Color::Black);
            b.place(pt7(2, c), Color::Black);
        }
        b.place(pt7(1, 0), Color::Black);
        b.place(pt7(1, 2), Color::Black);
        b.place(pt7(1, 4), Color::Black);

        for c in 2..7 {
            b.place(pt7(4, c), Color::White);
            b.place(pt7(6, c), Color::White);
        }
        b.place(pt7(5, 2), Color::White);
        b.place(pt7(5, 4), Color::White);
        b.place(pt7(5, 6), Color::White);

        let life = pure_eye_analysis(&b);
        assert!(!life.black_alive.is_empty());
        assert!(!life.white_alive.is_empty());
        assert!(life.black_alive.get(pt7(0, 0).index()));
        assert!(life.white_alive.get(pt7(6, 6).index()));
    }

    #[test]
    fn alive_group_counts_eyes_as_territory() {
        let mut b = B5x5::new();
        for c in 0..5 {
            b.place(pt5(0, c), Color::Black);
            b.place(pt5(2, c), Color::Black);
        }
        b.place(pt5(1, 0), Color::Black);
        b.place(pt5(1, 2), Color::Black);
        b.place(pt5(1, 4), Color::Black);

        let life = pure_eye_analysis(&b);
        assert!(life.black_territory.get(pt5(1, 1).index()));
        assert!(life.black_territory.get(pt5(1, 3).index()));
        assert!(life.white_territory.is_empty());
        assert!(!life.black_territory.get(pt5(3, 0).index()));
    }

    #[test]
    fn three_by_three_corner_not_alive() {
        // 3x3 ring with single connected eye-region: not alive.
        let mut b = B3x3::new();
        for c in 0..3 {
            b.place(Point::<3, 3>::new(0, c).unwrap(), Color::Black);
        }
        b.place(Point::<3, 3>::new(1, 0).unwrap(), Color::Black);
        b.place(Point::<3, 3>::new(1, 2).unwrap(), Color::Black);
        b.place(Point::<3, 3>::new(2, 0).unwrap(), Color::Black);
        b.place(Point::<3, 3>::new(2, 2).unwrap(), Color::Black);
        let life = pure_eye_analysis(&b);
        assert!(life.black_alive.is_empty());
    }
}
