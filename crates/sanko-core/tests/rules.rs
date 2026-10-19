//! Integration tests for the PSK rules engine.
//!
//! Covers captures, suicide, PSK rejection (an immediate ko recapture, and a
//! longer repetition that basic ko would allow), and play/unmake hash-history
//! reversibility.

use sanko_core::{
    BitBoard, Color, HashHistory, Move, PlayError, Point, Position3x3, Position4x4,
    define_position, terminal_winner,
};

fn stone<const R: usize, const C: usize>(row: usize, col: usize) -> Move<R, C> {
    Move::Stone(Point::<R, C>::new(row, col).unwrap())
}

fn caps_rc<const R: usize, const C: usize, const L: usize>(
    bb: &BitBoard<L>,
) -> Vec<(usize, usize)> {
    bb.iter()
        .map(|idx| {
            let p = Point::<R, C>::from_index_unchecked(idx);
            (p.row(), p.col())
        })
        .collect()
}

#[test]
fn empty_position_defaults() {
    let p = Position3x3::new();
    let h = HashHistory::new();
    assert_eq!(p.to_move(), Color::Black);
    assert!(!p.is_terminal());
    assert!(p.legal(Move::Pass, &h));
}

#[test]
fn pass_pass_terminates() {
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    p.play(Move::Pass, &mut h).unwrap();
    assert!(!p.is_terminal());
    p.play(Move::Pass, &mut h).unwrap();
    assert!(p.is_terminal());
}

#[test]
fn simple_capture_single_stone() {
    // Surround the center stone, capture it.
    //   . X .
    //   X O X      Black plays (1,2) -> captures W at (1,1)
    //   . X .
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    p.play(stone::<3, 3>(0, 1), &mut h).unwrap(); // B
    p.play(stone::<3, 3>(1, 1), &mut h).unwrap(); // W
    p.play(stone::<3, 3>(1, 0), &mut h).unwrap(); // B
    p.play(Move::Pass, &mut h).unwrap(); // W passes
    p.play(stone::<3, 3>(2, 1), &mut h).unwrap(); // B
    p.play(Move::Pass, &mut h).unwrap(); // W passes
    let undo = p.play(stone::<3, 3>(1, 2), &mut h).unwrap(); // B captures
    assert_eq!(caps_rc::<3, 3, _>(&undo.captures), vec![(1, 1)]);
    assert_eq!(p.board().at(Point::<3, 3>::new(1, 1).unwrap()), None);
}

#[test]
fn suicide_rejected() {
    //  . X .
    //  X . .       After B(0,1) and B(1,0), W trying (0,0) is suicide.
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    p.play(stone::<3, 3>(0, 1), &mut h).unwrap(); // B
    p.play(Move::Pass, &mut h).unwrap(); // W
    p.play(stone::<3, 3>(1, 0), &mut h).unwrap(); // B
    // Now W to move tries (0, 0), which is suicide.
    assert_eq!(p.play(stone::<3, 3>(0, 0), &mut h), Err(PlayError::Suicide));
    // Position not mutated, history unchanged. Two stone moves pushed (the
    // pass between them does not push).
    assert_eq!(h.len(), 2);
    assert_eq!(p.to_move(), Color::White);
}

#[test]
fn suicide_with_capture_is_legal() {
    // Classic capture-by-suicide shape on 3x3:
    //   X O O           B chain in column 0 has its last liberty at (1,1).
    //   X . O           W plays (1,1). Without the capture, W's chain would
    //   X O O           be suicide (zero libs). With capture, W gains
    //                   libs (0,0), (1,0), (2,0).
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    for mv in [
        stone::<3, 3>(0, 0), // B
        stone::<3, 3>(0, 1), // W
        stone::<3, 3>(1, 0), // B
        stone::<3, 3>(0, 2), // W
        stone::<3, 3>(2, 0), // B
        stone::<3, 3>(1, 2), // W
        Move::Pass,          // B
        stone::<3, 3>(2, 1), // W
        Move::Pass,          // B
        stone::<3, 3>(2, 2), // W
        Move::Pass,          // B (W to play)
    ] {
        p.play(mv, &mut h).unwrap();
    }
    let undo = p.play(stone::<3, 3>(1, 1), &mut h).unwrap();
    let captured: std::collections::HashSet<(usize, usize)> =
        caps_rc::<3, 3, _>(&undo.captures).into_iter().collect();
    assert_eq!(
        captured,
        [(0, 0), (1, 0), (2, 0)].iter().copied().collect(),
        "should capture the three B stones along column 0"
    );
    assert_eq!(
        p.board().at(Point::<3, 3>::new(1, 1).unwrap()),
        Some(Color::White)
    );
}

#[test]
fn psk_rejects_immediate_ko_recapture() {
    // Classic ko on 4x4.
    //   . X O .
    //   X . X O      After all setup, W can capture B(1,2) via W(1,1).
    //   . X O .      Then B retaking at (1,2) immediately is PSK violation.
    //   . . . .
    let mut p = Position4x4::new();
    let mut h = HashHistory::new();
    for mv in [
        stone::<4, 4>(0, 1),
        stone::<4, 4>(0, 2),
        stone::<4, 4>(1, 0),
        stone::<4, 4>(1, 3),
        stone::<4, 4>(2, 1),
        stone::<4, 4>(2, 2),
        stone::<4, 4>(1, 2),
    ] {
        p.play(mv, &mut h).unwrap();
    }
    // W captures at (1, 1).
    let undo = p.play(stone::<4, 4>(1, 1), &mut h).unwrap();
    assert_eq!(caps_rc::<4, 4, _>(&undo.captures), vec![(1, 2)]);
    // B tries immediate ko recapture at (1, 2). That would restore the prior
    // board, so PSK rejects.
    assert_eq!(
        p.play(stone::<4, 4>(1, 2), &mut h),
        Err(PlayError::PositionalSuperko)
    );
}

#[test]
fn psk_does_not_block_after_ko_threat() {
    // Same ko shape: after W captures and B plays an unrelated move,
    // and W plays an unrelated move, B can now play (1,2) again. The
    // resulting board is *different* from any prior because of the
    // intervening stones.
    let mut p = Position4x4::new();
    let mut h = HashHistory::new();
    for mv in [
        stone::<4, 4>(0, 1),
        stone::<4, 4>(0, 2),
        stone::<4, 4>(1, 0),
        stone::<4, 4>(1, 3),
        stone::<4, 4>(2, 1),
        stone::<4, 4>(2, 2),
        stone::<4, 4>(1, 2),
        stone::<4, 4>(1, 1), // W captures B(1,2)
    ] {
        p.play(mv, &mut h).unwrap();
    }
    // B plays elsewhere and adds a new stone, so the board is now different
    // from any seen before.
    p.play(stone::<4, 4>(3, 0), &mut h).unwrap();
    p.play(stone::<4, 4>(3, 1), &mut h).unwrap();
    // B can now play (1,2), which captures W(1,1). The resulting board is
    // distinct from any prior (the extra stones at (3,0)/(3,1) ensure that).
    p.play(stone::<4, 4>(1, 2), &mut h).expect("legal");
}

#[test]
fn play_unmake_restores_position_and_hash() {
    let mut p = Position4x4::new();
    let mut h = HashHistory::new();
    let original_hash = p.board().hash();
    let undo = p.play(stone::<4, 4>(1, 1), &mut h).unwrap();
    assert_ne!(p.board().hash(), original_hash);
    assert_eq!(h.len(), 1);
    p.unmake(&undo, &mut h);
    assert_eq!(p.board().hash(), original_hash);
    assert!(h.is_empty());
}

#[test]
fn occupied_rejection() {
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    p.play(stone::<3, 3>(1, 1), &mut h).unwrap();
    assert_eq!(
        p.play(stone::<3, 3>(1, 1), &mut h),
        Err(PlayError::Occupied)
    );
}

#[test]
fn terminal_winner_empty_board_with_komi() {
    // Empty board, pass-pass terminal. No stones, no regions counted
    // (whole-board empty region touches neither color). Black=0, White=0+komi.
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    p.play(Move::Pass, &mut h).unwrap();
    p.play(Move::Pass, &mut h).unwrap();
    assert!(p.is_terminal());
    // With 8.5 komi (= 17 half-points), W wins by 8.5.
    assert_eq!(terminal_winner(&p, 17), Some(Color::White));
    // With komi -0.5 (-1 half-points), B wins by half a point.
    assert_eq!(terminal_winner(&p, -1), Some(Color::Black));
}

// A repetition that basic ko allows but positional superko forbids. On 1×5
// the board `. . O X .` from the second move of the game would come back nine
// plies later. Basic ko only forbids recreating the board from just before the
// opponent's last move (here `X X . . .`), so it would allow the move.
define_position!(P1x5, 1, 5);

#[test]
fn psk_rejects_repetition_beyond_basic_ko() {
    let mut p = P1x5::new();
    let mut h = HashHistory::new();
    // B D1, W C1 (`. . O X .`), B pass, W E1 takes D1, B pass, W D1, B B1,
    // W pass, B A1 takes C1-E1 (`X X . . .`), W C1 takes A1-B1 (`. . O . .`).
    let col = |c: usize| stone::<1, 5>(0, c);
    let moves = [col(3), col(2), Move::Pass, col(4), Move::Pass, col(3), col(1), Move::Pass, col(0), col(2)];
    let mut boards = Vec::new();
    for m in moves {
        p.play(m, &mut h).unwrap();
        boards.push(p.board().hash());
    }
    assert_eq!(p.to_move(), Color::Black);
    // Black D1 would recreate the board after move 2, which is not the board
    // from just before White's last move: a long cycle, not a basic ko.
    let d1 = Point::<1, 5>::new(0, 3).unwrap();
    assert_eq!(p.board_hash_after(d1), Some(boards[1]));
    assert_ne!(boards[1], boards[moves.len() - 2]);
    assert!(!p.legal(Move::Stone(d1), &h));
    assert_eq!(p.play(Move::Stone(d1), &mut h).err(), Some(PlayError::PositionalSuperko));
    // The position is unchanged and other moves stay legal.
    assert_eq!(p.board().hash(), boards[moves.len() - 1]);
    assert!(p.legal(col(0), &h));
}
