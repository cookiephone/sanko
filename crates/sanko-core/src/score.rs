//! Terminal winner and area (Chinese-style) scoring.
//!
//! At a terminal position (two consecutive passes), the score is computed as:
//! - each color's stone count, plus
//! - each empty region fully bordered by one color, attributed to that color.
//!
//! No automatic dead-stone removal: under PSK the game is played out, and
//! any stones remaining at the terminal are alive by definition.
//!
//! Komi is applied as a White bonus: `white_score = white_area + komi`.
//! Half-point komi prevents draws and is the standard solver convention.

use core::cmp::Ordering;
use core::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

use crate::bitboard::BitBoard;
use crate::color::Color;
use crate::point::Point;
use crate::position::Position;
use crate::topology::dilate;

/// Experimental rule variant: MIGOS II's end-of-game convention (van der Werf &
/// Winands 2009, §3.2). At a terminal position, chains with exactly one liberty
/// are dead (removed before counting) if the opponent has no such chains. NOT
/// the rules of our results, only for explaining value differences to MIGOS.
static ATARI_DEAD: AtomicBool = AtomicBool::new(false);

/// Enable the experimental atari-dead terminal convention (see `ATARI_DEAD`).
pub fn set_atari_dead(on: bool) {
    ATARI_DEAD.store(on, AtomicOrdering::Relaxed);
}

/// Area-score `(black, white)` at any position. Counts stones plus
/// single-color empty regions.
#[must_use]
pub fn area_score<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
>(
    pos: &Position<ROWS, COLS, LANES, NPOINTS>,
) -> (u32, u32) {
    let board = pos.board();
    area_of::<ROWS, COLS, LANES, NPOINTS>(
        board.stones(Color::Black),
        board.stones(Color::White),
        board.empties(),
    )
}

/// Union of the chains of `stones` that have exactly one liberty.
fn atari_chains<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    stones: BitBoard<LANES>,
    empties: BitBoard<LANES>,
) -> BitBoard<LANES> {
    let mut atari: BitBoard<LANES> = BitBoard::EMPTY;
    let mut rest = stones;
    while let Some(i) = rest.iter().next() {
        let mut chain: BitBoard<LANES> = BitBoard::singleton(i);
        loop {
            let grown = dilate::<ROWS, COLS, LANES, NPOINTS>(&chain).intersect(&stones);
            if grown == chain {
                break;
            }
            chain = grown;
        }
        rest = rest.difference(&chain);
        if dilate::<ROWS, COLS, LANES, NPOINTS>(&chain).intersect(&empties).count() == 1 {
            atari = atari.union(&chain);
        }
    }
    atari
}

fn area_of<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    black_stones: BitBoard<LANES>,
    white_stones: BitBoard<LANES>,
    empties_bb: BitBoard<LANES>,
) -> (u32, u32) {
    let mut visited: BitBoard<LANES> = BitBoard::EMPTY;
    let mut black = black_stones.count();
    let mut white = white_stones.count();

    for r in 0..ROWS {
        for c in 0..COLS {
            let p = Point::<ROWS, COLS>::new_unchecked(r, c);
            let i = p.index();
            if !empties_bb.get(i) || visited.get(i) {
                continue;
            }
            // Flood-fill the empty region starting at `p` by repeated
            // dilate ∩ empties.
            let mut region: BitBoard<LANES> = BitBoard::EMPTY;
            region.set(i);
            loop {
                let grown =
                    dilate::<ROWS, COLS, LANES, NPOINTS>(&region).intersect(&empties_bb);
                if grown == region {
                    break;
                }
                region = grown;
            }
            visited = visited.union(&region);
            let region_size = region.count();
            let dilated = dilate::<ROWS, COLS, LANES, NPOINTS>(&region);
            let touches_black = !dilated.intersect(&black_stones).is_empty();
            let touches_white = !dilated.intersect(&white_stones).is_empty();
            match (touches_black, touches_white) {
                (true, false) => black += region_size,
                (false, true) => white += region_size,
                _ => {}
            }
        }
    }
    (black, white)
}

/// Determine the winner of a terminal position, given komi (White's bonus).
/// `komi_times_two` is twice the komi, an integer, so the comparison of
/// `2·black` with `2·white + komi_times_two` is exact. Returns `None` only at
/// draws, which are impossible with half-integer komi.
#[must_use]
pub fn terminal_winner<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
>(
    pos: &Position<ROWS, COLS, LANES, NPOINTS>,
    komi_times_two: i32,
) -> Option<Color> {
    let (b, w) = if ATARI_DEAD.load(AtomicOrdering::Relaxed) {
        let board = pos.board();
        let (mut black, mut white, mut empties) =
            (board.stones(Color::Black), board.stones(Color::White), board.empties());
        let ba = atari_chains::<ROWS, COLS, LANES, NPOINTS>(black, empties);
        let wa = atari_chains::<ROWS, COLS, LANES, NPOINTS>(white, empties);
        if ba.is_empty() && !wa.is_empty() {
            white = white.difference(&wa);
            empties = empties.union(&wa);
        } else if wa.is_empty() && !ba.is_empty() {
            black = black.difference(&ba);
            empties = empties.union(&ba);
        }
        area_of::<ROWS, COLS, LANES, NPOINTS>(black, white, empties)
    } else {
        area_score(pos)
    };
    // Compare `2*black` vs `2*white + 2*komi = 2*white + komi_times_two`.
    let lhs = (b as i32) * 2;
    let rhs = (w as i32) * 2 + komi_times_two;
    match lhs.cmp(&rhs) {
        Ordering::Greater => Some(Color::Black),
        Ordering::Less => Some(Color::White),
        Ordering::Equal => None,
    }
}
