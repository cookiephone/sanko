//! The experimental MIGOS end-of-game convention (`set_atari_dead`). Its own
//! test binary: the switch is process-global.

use sanko_core::{Color, HashHistory, Move, Point, Position3x3, set_atari_dead, terminal_winner};

const fn stone(row: usize, col: usize) -> Move<3, 3> {
    Move::Stone(Point::<3, 3>::new(row, col).unwrap())
}

#[test]
fn white_stone_in_atari_is_dead_only_under_the_convention() {
    let mut p = Position3x3::new();
    let mut h = HashHistory::new();
    for mv in [stone(1, 1), stone(0, 0), stone(0, 1), Move::Pass, Move::Pass] {
        p.play(mv, &mut h).unwrap();
    }
    assert!(p.is_terminal());
    // Plain area: Black 2, White 1 (the remaining empty region touches both).
    // Komi 3.5: 4 < 2 + 7 half-points → White.
    set_atari_dead(false);
    assert_eq!(terminal_winner(&p, 7), Some(Color::White));
    // White's (0,0) has one liberty and Black has none in atari: it is removed,
    // and the whole empty area becomes Black's (9 points).
    set_atari_dead(true);
    assert_eq!(terminal_winner(&p, 7), Some(Color::Black));
    assert_eq!(terminal_winner(&p, 17), Some(Color::Black));
    assert_eq!(terminal_winner(&p, 19), Some(Color::White));
    set_atari_dead(false);
}
