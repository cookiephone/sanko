//! df-pn solver regression tests: known small-board values (lower and upper
//! bounds), and a differential check of the GHI proof-transfer machinery and
//! the closed-region solver against the exact-only oracle (proof reuse only
//! on the identical superko game, no region solver).

use sanko_core::{Color, HashHistory, Move, Point, Position, Rng};
use sanko_search::{DfpnConfig, solve_dfpn};

fn cfg(komi_times_two: i32) -> DfpnConfig {
    DfpnConfig {
        komi_times_two,
        tt_log_buckets: 16,
        static_cache_log_slots: 16,
        ..DfpnConfig::default()
    }
}

fn winner<const R: usize, const C: usize, const L: usize, const N: usize>(c: DfpnConfig) -> Option<Color> {
    solve_dfpn::<R, C, L, N>(&[], c).winner
}

#[test]
fn known_values_small_boards() {
    // Area scoring, positional superko: 2×2 B+1, 2×3 0, 3×3 B+9, 3×4 B+4.
    assert_eq!(winner::<2, 2, 1, 4>(cfg(1)), Some(Color::Black));
    assert_eq!(winner::<2, 2, 1, 4>(cfg(3)), Some(Color::White));
    assert_eq!(winner::<2, 3, 1, 6>(cfg(1)), Some(Color::White));
    assert_eq!(winner::<2, 3, 1, 6>(cfg(-1)), Some(Color::Black));
    assert_eq!(winner::<3, 3, 1, 9>(cfg(17)), Some(Color::Black));
    assert_eq!(winner::<3, 3, 1, 9>(cfg(19)), Some(Color::White));
    assert_eq!(winner::<3, 4, 1, 12>(cfg(7)), Some(Color::Black));
    assert_eq!(winner::<3, 4, 1, 12>(cfg(9)), Some(Color::White));
}

#[test]
fn four_by_four_is_black_by_two() {
    let c = DfpnConfig { tt_log_buckets: 20, ..cfg(3) };
    assert_eq!(winner::<4, 4, 1, 16>(c), Some(Color::Black));
    assert_eq!(winner::<4, 4, 1, 16>(DfpnConfig { komi_times_two: 5, threads: 4, ..c }), Some(Color::White));
}

#[test]
fn kill_all_configuration_proves_full_board_win() {
    // 3×5 is a full-board Black win (B+15): the conjectured-winner pruning,
    // kill-all prior and closed-region solver must still find the proof...
    let c = DfpnConfig {
        prune_for: Some(Color::Black),
        eye_init: 8,
        eye_area: true,
        eye_weight: 4,
        region_solver: true,
        tt_log_buckets: 18,
        ..cfg(29)
    };
    assert_eq!(winner::<3, 5, 1, 15>(c), Some(Color::Black));
    // ...and a searched result against the pruned side is never reported (3×4
    // at komi 4.5 is a White win, but not statically decided at the root).
    assert_eq!(winner::<3, 4, 1, 12>(DfpnConfig { komi_times_two: 9, ..c }), None);
    assert_eq!(winner::<3, 4, 1, 12>(DfpnConfig { komi_times_two: 7, ..c }), Some(Color::Black));
}

/// Two 1×5 games with the same set of boards that end on mirror images of
/// each other (`..OOO` and `OOO..`, Black to move). They are different superko
/// games with different winners at komi -2.5 (brute-force minimax): a proof
/// stored for one must never be reused for the other.
#[test]
fn mirror_endpoints_with_the_same_board_set_are_different_games() {
    let line = |s: &str| -> Vec<Move<1, 5>> {
        s.split_whitespace()
            .map(|t| if t == "P" { Move::Pass } else { Move::Stone(Point::<1, 5>::new(0, t.parse().unwrap()).unwrap()) })
            .collect()
    };
    let a = line("1 P 0 2 0 1 P 0 3 2 P 4 P 3");
    let b = line("3 2 P 4 P 3 1 P 0 2 0 1 P 0");
    let c = cfg(-5);
    assert_eq!(solve_dfpn::<1, 5, 1, 5>(&a, c).winner, Some(Color::White));
    assert_eq!(solve_dfpn::<1, 5, 1, 5>(&b, c).winner, Some(Color::Black));
    let exact = DfpnConfig { exact_only: true, no_static: true, simulate: 0, ..c };
    assert_eq!(solve_dfpn::<1, 5, 1, 5>(&a, exact).winner, Some(Color::White));
    assert_eq!(solve_dfpn::<1, 5, 1, 5>(&b, exact).winner, Some(Color::Black));
}

/// Random legal prefix of up to `max_len` moves (real superko history).
fn random_prefix<const R: usize, const C: usize, const L: usize, const N: usize>(
    rng: &mut Rng,
    max_len: u32,
) -> Option<Vec<Move<R, C>>> {
    let mut pos: Position<R, C, L, N> = Position::new();
    let mut hist = HashHistory::new();
    hist.push(pos.board().key());
    let mut moves = Vec::new();
    for _ in 0..rng.next_below(max_len) {
        let mut legal: Vec<Move<R, C>> = (0..N)
            .map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k)))
            .filter(|&m| pos.legal(m, &hist))
            .collect();
        if legal.is_empty() || rng.next_below(10) == 0 {
            legal.push(Move::Pass);
        }
        let mv = legal[rng.next_below(legal.len() as u32) as usize];
        pos.play(mv, &mut hist).ok()?;
        moves.push(mv);
        if pos.consecutive_passes() > 0 {
            break;
        }
    }
    Some(moves)
}

#[test]
fn transfers_and_regions_agree_with_exact_only_oracle() {
    let mut rng = Rng::new(0x5EED);
    let mut compared = 0;
    for _ in 0..60 {
        let Some(moves) = random_prefix::<3, 4, 1, 12>(&mut rng, 14) else { continue };
        let komi_times_two = 2 * (rng.next_below(13) as i32 - 6) + 1;
        let full = DfpnConfig { region_solver: true, node_limit: 2_000_000, ..cfg(komi_times_two) };
        let oracle = DfpnConfig { exact_only: true, node_limit: 2_000_000, simulate: 0, ..cfg(komi_times_two) };
        let a = solve_dfpn::<3, 4, 1, 12>(&moves, full).winner;
        let b = solve_dfpn::<3, 4, 1, 12>(&moves, oracle).winner;
        if let (Some(a), Some(b)) = (a, b) {
            assert_eq!(a, b, "moves={moves:?} komi2={komi_times_two}");
            compared += 1;
        }
    }
    assert!(compared > 40, "only {compared} positions compared");
}
