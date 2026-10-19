//! Chain-walk helpers for the PSK rules engine.
//!
//! These operate on `Board` directly (no `Position`) so the rules layer can
//! probe a hypothetical board (e.g. inside the suicide check before
//! committing the move).

use crate::bitboard::BitBoard;
use crate::board::Board;
use crate::color::Color;
use crate::point::Point;
use crate::topology::dilate;

/// Flood the `color` chain from `seed` (a set of its stones) until it either
/// touches an empty point (returns `(stones flooded so far, true)`) or is
/// complete without liberties (returns `(whole chain, false)`). A liberty of a
/// partially flooded chain is a liberty of the chain, so stopping at the first
/// one is exact. Most chains have one next to their first stones.
#[inline]
fn flood_until_liberty<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
    seed: BitBoard<LANES>,
    color: Color,
) -> (BitBoard<LANES>, bool) {
    let same = board.stones(color);
    let empties = board.empties();
    let mut chain = seed;
    loop {
        let grown = dilate::<ROWS, COLS, LANES, NPOINTS>(&chain);
        if !grown.intersect(&empties).is_empty() {
            return (chain, true);
        }
        let next = grown.intersect(&same);
        if next == chain {
            return (chain, false);
        }
        chain = next;
    }
}

/// Whether the chain at `p` (an occupied point) has at least one liberty.
/// Flood-fills the chain and exits at the first liberty found.
pub(crate) fn chain_has_liberty<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    board: &Board<ROWS, COLS, LANES, NPOINTS>,
    p: Point<ROWS, COLS>,
) -> bool {
    let color = board.at(p).expect("chain_has_liberty called on empty point");
    flood_until_liberty(board, BitBoard::singleton(p.index()), color).1
}

/// Resolve captures around a freshly placed stone at `p` (color `mover`).
/// Removes captured opponent chains from `board` in-place and returns the
/// bitboard of removed stones. The chain at `p` is **not** checked for
/// suicide here. Do that after this call.
///
/// An adjacent opponent chain is captured iff it has no liberty. Each chain
/// is flooded only until its first liberty, and stones already examined (in a
/// captured chain or a partial flood that found a liberty) are skipped. The
/// order of the neighbours does not matter: two opponent chains next to `p`
/// cannot touch each other (they would be one chain), so removing one never
/// gives the other a liberty.
pub(crate) fn resolve_captures<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
>(
    board: &mut Board<ROWS, COLS, LANES, NPOINTS>,
    p: Point<ROWS, COLS>,
    mover: Color,
) -> BitBoard<LANES> {
    let opp = mover.opponent();
    let mut captures: BitBoard<LANES> = BitBoard::EMPTY;
    let mut examined: BitBoard<LANES> = BitBoard::EMPTY;
    for n in p.neighbors().into_iter().flatten() {
        if examined.get(n.index()) || board.at(n) != Some(opp) {
            continue;
        }
        let (chain, has_liberty) = flood_until_liberty(board, BitBoard::singleton(n.index()), opp);
        examined = examined.union(&chain);
        if !has_liberty {
            for idx in &chain {
                let cp = Point::<ROWS, COLS>::from_index_unchecked(idx);
                board.remove(cp, opp);
            }
            captures = captures.union(&chain);
        }
    }
    captures
}
