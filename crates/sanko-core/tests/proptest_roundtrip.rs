//! Property tests: play/unmake round-trips fully restore Position + history.

use sanko_core::{HashHistory, Move, Point, Position4x4};
use proptest::prelude::*;

fn arb_move() -> impl Strategy<Value = Move<4, 4>> {
    prop_oneof![
        Just(Move::Pass),
        (0usize..4, 0usize..4).prop_map(|(r, c)| Move::Stone(Point::<4, 4>::new(r, c).unwrap())),
    ]
}

proptest! {
    /// Play a random legal-or-illegal sequence of moves and track the
    /// position+history snapshot before each successful play. Then
    /// unmake in reverse and verify each snapshot is restored.
    #[test]
    fn play_unmake_round_trips_on_4x4(
        moves in proptest::collection::vec(arb_move(), 0..40)
    ) {
        let mut p = Position4x4::new();
        let mut h = HashHistory::new();
        let mut undos = Vec::new();
        let mut snapshots = Vec::new();

        for mv in moves {
            snapshots.push((p.clone(), h.clone()));
            match p.play(mv, &mut h) {
                Ok(undo) => undos.push(undo),
                Err(_) => {
                    snapshots.pop();
                }
            }
        }

        while let Some(undo) = undos.pop() {
            let (expected_p, expected_h) = snapshots.pop().unwrap();
            p.unmake(&undo, &mut h);
            prop_assert_eq!(p.board().hash(), expected_p.board().hash());
            prop_assert_eq!(p.to_move(), expected_p.to_move());
            prop_assert_eq!(p.consecutive_passes(), expected_p.consecutive_passes());
            prop_assert_eq!(h.len(), expected_h.len());
        }
        prop_assert_eq!(h.len(), 0);
        prop_assert_eq!(p.board().hash(), 0);
    }
}
