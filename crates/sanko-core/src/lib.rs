//! Board representation and rules engine for the Sanko Go solver: the Chinese
//! rules as written, with positional superko (PSK), played out to the end.
//!
//! Boards are `ROWS × COLS` (rectangular). Square is just the special case
//! where `ROWS == COLS`. Use the [`define_board!`] / [`define_position!`]
//! macros or the pre-defined type aliases (`Board5x5`, `Position5x5`, …)
//! rather than naming `Board` / `Position` directly. They bind the four
//! const generics consistently.
//!
//! The API serves the `sanko` solver and is not stable: it may change in any
//! 0.x release.

mod bitboard;
mod board;
mod color;
mod history;
mod mv;
mod point;
mod position;
mod pure_eye;
mod rng;
mod rules;
mod safety;
mod score;
mod symmetry;
mod topology;
mod zobrist;

pub use bitboard::BitBoard;
pub use board::Board;
pub use color::Color;
pub use history::HashHistory;
pub use mv::Move;
pub use point::Point;
pub use position::{PlayError, Position, UndoInfo};
pub use pure_eye::{PureEyeAnalysis, pure_eye_analysis, pure_eye_analysis_of, secure_upper_bound};
pub use rng::Rng;
pub use safety::{may_be_decided, statically_decided, statically_decided_life};
pub use score::{area_score, atari_dead, set_atari_dead, terminal_winner};
pub use symmetry::Symmetry;
pub use zobrist::{PASS_KEY, STM_BLACK, STM_WHITE};

/// Orthogonal neighbours of `v` (raw bits of a board of at most 64 points).
#[inline]
#[must_use]
pub const fn neighbors_bits<const COLS: usize, const NPOINTS: usize>(v: u64) -> u64 {
    topology::neighbors_bits::<COLS, NPOINTS>(v)
}

/// `v` ∪ its orthogonal neighbours (raw bits of a board of at most 64 points).
#[inline]
#[must_use]
pub const fn dilate_bits<const COLS: usize, const NPOINTS: usize>(v: u64) -> u64 {
    v | topology::neighbors_bits::<COLS, NPOINTS>(v)
}

/// `b` ∪ its orthogonal neighbours (4-connected dilation), within the board.
#[inline]
#[must_use]
pub fn dilate4<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    b: &BitBoard<LANES>,
) -> BitBoard<LANES> {
    topology::dilate::<ROWS, COLS, LANES, NPOINTS>(b)
}

/// Declare a `Board` type alias for a specific `ROWS × COLS` size, computing
/// the `LANES` and `NPOINTS` const-generic parameters automatically.
#[macro_export]
macro_rules! define_board {
    ($vis:vis $alias:ident, $rows:expr, $cols:expr) => {
        $vis type $alias = $crate::Board<
            { $rows },
            { $cols },
            {
                let n: usize = ($rows) * ($cols);
                n.div_ceil(64)
            },
            { ($rows) * ($cols) },
        >;
    };
}

/// Declare a `Position` type alias for a specific `ROWS × COLS` size.
#[macro_export]
macro_rules! define_position {
    ($vis:vis $alias:ident, $rows:expr, $cols:expr) => {
        $vis type $alias = $crate::Position<
            { $rows },
            { $cols },
            {
                let n: usize = ($rows) * ($cols);
                n.div_ceil(64)
            },
            { ($rows) * ($cols) },
        >;
    };
}

define_board!(pub Board1x1, 1, 1);
define_board!(pub Board1x2, 1, 2);
define_board!(pub Board2x2, 2, 2);
define_board!(pub Board2x3, 2, 3);
define_board!(pub Board3x3, 3, 3);
define_board!(pub Board4x4, 4, 4);
define_board!(pub Board5x5, 5, 5);
define_board!(pub Board6x6, 6, 6);
define_board!(pub Board7x7, 7, 7);

define_position!(pub Position1x1, 1, 1);
define_position!(pub Position1x2, 1, 2);
define_position!(pub Position2x2, 2, 2);
define_position!(pub Position2x3, 2, 3);
define_position!(pub Position3x3, 3, 3);
define_position!(pub Position4x4, 4, 4);
define_position!(pub Position5x5, 5, 5);
define_position!(pub Position6x6, 6, 6);
define_position!(pub Position7x7, 7, 7);
