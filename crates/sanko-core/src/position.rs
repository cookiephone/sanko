//! Rules engine on top of `Board`: the Chinese rules as written, whose ban on
//! repeating a whole-board position is positional superko (PSK).
//!
//! `Position` = `Board` + side-to-move + consecutive-pass counter. PSK history
//! is **not** stored here. It lives in a separate [`crate::HashHistory`]
//! object owned by the search path, so that transposition-table keys remain
//! history-free and the GHI problem is handled explicitly at the TT layer.
//!
//! Rules:
//! - **No suicide**: the placed stone's chain must have ≥1 liberty after
//!   resolving captures. Suicidal moves are illegal.
//! - **Positional superko**: a move that produces a whole-board position
//!   (color configuration) equal to any earlier position on the search path
//!   is illegal. Checked against the supplied `HashHistory`.
//! - **No basic ko separately**: PSK strictly supersedes basic ko.
//! - **Area scoring**: at terminal (two consecutive passes), the score is
//!   `stones(C) + empty_regions_touching_only(C)`. Dead stones are not agreed
//!   on: the game is played out, and any stones remaining at the terminal are
//!   alive by definition. Under area scoring playing on costs neither side
//!   anything, so this changes no value.

use core::fmt;

use crate::bitboard::BitBoard;
use crate::board::Board;
use crate::color::Color;
use crate::history::HashHistory;
use crate::mv::Move;
use crate::point::Point;
use crate::rules::{chain_has_liberty, resolve_captures};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PlayError {
    Occupied,
    Suicide,
    PositionalSuperko,
}

/// State needed to reverse a `play()` call. `pushed_hash = Some(h)` means
/// `play` pushed hash `h` onto history (always true for stone moves).
/// `None` means `play` did not push (always for passes: since they don't
/// change the board, they don't add to PSK history). `unmake` pops iff
/// `pushed_hash` is Some.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct UndoInfo<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
> {
    prev_board: Board<ROWS, COLS, LANES, NPOINTS>,
    prev_to_move: Color,
    prev_consecutive_passes: u8,
    pushed_hash: Option<u64>,
    /// Stones removed by this move. Empty bitboard if no captures (or for
    /// passes).
    pub captures: BitBoard<LANES>,
}

#[derive(Clone, Eq, PartialEq)]
pub struct Position<
    const ROWS: usize,
    const COLS: usize,
    const LANES: usize,
    const NPOINTS: usize,
> {
    board: Board<ROWS, COLS, LANES, NPOINTS>,
    to_move: Color,
    consecutive_passes: u8,
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>
    Position<ROWS, COLS, LANES, NPOINTS>
{
    #[must_use]
    pub const fn new() -> Self {
        Self {
            board: Board::new(),
            to_move: Color::Black,
            consecutive_passes: 0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn board(&self) -> &Board<ROWS, COLS, LANES, NPOINTS> {
        &self.board
    }

    #[inline]
    #[must_use]
    pub const fn to_move(&self) -> Color {
        self.to_move
    }

    /// Two consecutive passes: the game has ended.
    #[inline]
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.consecutive_passes >= 2
    }

    #[inline]
    #[must_use]
    pub const fn consecutive_passes(&self) -> u8 {
        self.consecutive_passes
    }

    #[inline]
    #[must_use]
    pub const fn stone_count(&self) -> u32 {
        self.board.stone_count()
    }

    /// Iterator over empty intersections in row-major index order (the
    /// stone-move candidates).
    pub fn empty_points(&self) -> impl Iterator<Item = Point<ROWS, COLS>> + '_ {
        self.board
            .empties()
            .iter()
            .map(Point::<ROWS, COLS>::from_index_unchecked)
    }

    /// Test whether a move is legal under the current position **and** the
    /// supplied PSK history. Does not mutate either.
    #[must_use]
    pub fn legal(&self, mv: Move<ROWS, COLS>, history: &HashHistory) -> bool {
        match mv {
            Move::Pass => true,
            Move::Stone(p) => self.simulate_stone(p, history).is_ok(),
        }
    }

    /// Board hash the stone move at `p` would produce, ignoring superko.
    /// `None` if the point is occupied or the move is suicide. Used by search
    /// code to identify *which* earlier board a superko-illegal move would
    /// repeat (the GHI exclusion target).
    #[must_use]
    pub fn board_hash_after(&self, p: Point<ROWS, COLS>) -> Option<u64> {
        if self.board.at(p).is_some() {
            return None;
        }
        let mut new_board = self.board;
        new_board.place(p, self.to_move);
        resolve_captures(&mut new_board, p, self.to_move);
        if !chain_has_liberty(&new_board, p) {
            return None;
        }
        Some(new_board.hash())
    }

    /// The captures and resulting board hash of the stone move at `p`, ignoring
    /// superko, or `None` if occupied or suicide. Non-mutating: lets a search
    /// compute child keys (e.g. to warm caches) without touching the history.
    #[must_use]
    pub fn preview_stone(&self, p: Point<ROWS, COLS>) -> Option<(BitBoard<LANES>, u64)> {
        if self.board.at(p).is_some() {
            return None;
        }
        let mut new_board = self.board;
        new_board.place(p, self.to_move);
        let captures = resolve_captures(&mut new_board, p, self.to_move);
        if !chain_has_liberty(&new_board, p) {
            return None;
        }
        Some((captures, new_board.hash()))
    }

    /// Apply a move, updating both `self` and `history`. On failure, neither
    /// is touched. On success, returns an `UndoInfo` that, paired with the
    /// same `history`, can reverse the move via [`Self::unmake`].
    ///
    /// # Errors
    /// Returns [`PlayError`] for occupied target, suicide, or PSK violation.
    pub fn play(
        &mut self,
        mv: Move<ROWS, COLS>,
        history: &mut HashHistory,
    ) -> Result<UndoInfo<ROWS, COLS, LANES, NPOINTS>, PlayError> {
        match mv {
            Move::Pass => Ok(self.do_pass(history)),
            Move::Stone(p) => self.play_stone(p, history),
        }
    }

    /// Reverse a previous successful `play`. The same `history` that was used
    /// in `play` must be passed. This is checked (in debug builds only) by the
    /// `pushed_hash` matching the top of the stack (only popped for stone moves).
    pub fn unmake(
        &mut self,
        undo: &UndoInfo<ROWS, COLS, LANES, NPOINTS>,
        history: &mut HashHistory,
    ) {
        if let Some(pushed) = undo.pushed_hash {
            let popped = history.pop();
            debug_assert_eq!(
                popped,
                Some(pushed),
                "history top did not match UndoInfo.pushed_hash — caller mixed up history objects"
            );
        }
        self.board = undo.prev_board;
        self.to_move = undo.prev_to_move;
        self.consecutive_passes = undo.prev_consecutive_passes;
    }

    /// Pass move. Does not change the board, so does not push to PSK history.
    /// Passes never violate PSK (the resulting board is identical to the
    /// pre-pass board, which is already on the path, so PSK as applied to
    /// stone moves can't be triggered by a pass).
    const fn do_pass(
        &mut self,
        _history: &mut HashHistory,
    ) -> UndoInfo<ROWS, COLS, LANES, NPOINTS> {
        let prev = UndoInfo {
            prev_board: self.board,
            prev_to_move: self.to_move,
            prev_consecutive_passes: self.consecutive_passes,
            pushed_hash: None,
            captures: BitBoard::EMPTY,
        };
        self.consecutive_passes = self.consecutive_passes.saturating_add(1);
        self.to_move = self.to_move.opponent();
        prev
    }

    /// Non-mutating stone-move legality probe.
    fn simulate_stone(
        &self,
        p: Point<ROWS, COLS>,
        history: &HashHistory,
    ) -> Result<(), PlayError> {
        if self.board.at(p).is_some() {
            return Err(PlayError::Occupied);
        }
        let mut new_board = self.board;
        new_board.place(p, self.to_move);
        resolve_captures(&mut new_board, p, self.to_move);
        if !chain_has_liberty(&new_board, p) {
            return Err(PlayError::Suicide);
        }
        if history.contains(new_board.hash()) {
            return Err(PlayError::PositionalSuperko);
        }
        Ok(())
    }

    fn play_stone(
        &mut self,
        p: Point<ROWS, COLS>,
        history: &mut HashHistory,
    ) -> Result<UndoInfo<ROWS, COLS, LANES, NPOINTS>, PlayError> {
        if self.board.at(p).is_some() {
            return Err(PlayError::Occupied);
        }
        let prev_board = self.board;
        let prev_to_move = self.to_move;
        let prev_consecutive_passes = self.consecutive_passes;

        self.board.place(p, self.to_move);
        let captures = resolve_captures(&mut self.board, p, self.to_move);

        if !chain_has_liberty(&self.board, p) {
            self.board = prev_board;
            return Err(PlayError::Suicide);
        }

        let new_hash = self.board.hash();
        if history.contains(new_hash) {
            self.board = prev_board;
            return Err(PlayError::PositionalSuperko);
        }

        self.consecutive_passes = 0;
        self.to_move = self.to_move.opponent();
        history.push(new_hash);

        Ok(UndoInfo {
            prev_board,
            prev_to_move,
            prev_consecutive_passes,
            pushed_hash: Some(new_hash),
            captures,
        })
    }
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> Default
    for Position<ROWS, COLS, LANES, NPOINTS>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> fmt::Debug
    for Position<ROWS, COLS, LANES, NPOINTS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Position<{ROWS}x{COLS}> to_move={:?} passes={}",
            self.to_move, self.consecutive_passes,
        )?;
        fmt::Display::fmt(&self.board, f)
    }
}
