//! Checkpoints: a solve that is stopped and resumed many times must reach the
//! same winner as an uninterrupted solve, serially and in parallel, a
//! checkpoint must be refused by a different problem and must not be
//! overwritten by a fresh run, a checkpoint seeds the runs of later positions
//! of its game (and only those), and runs at another komi, which must agree
//! with fresh solves there.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sanko_core::{Color, HashHistory, Move, Point, Position, Rng};
use sanko_search::{CheckpointError, CheckpointOptions, DfpnConfig, solve_dfpn, solve_dfpn_resumable};

fn cfg(komi_times_two: i32, threads: usize) -> DfpnConfig {
    DfpnConfig {
        komi_times_two,
        threads,
        tt_log_buckets: 17,
        proof_log: 16,
        static_cache_log_slots: 16,
        region_solver: true,
        region_interior: 8,
        ..DfpnConfig::default()
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("sanko-checkpoint-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(name);
    let _ = std::fs::remove_file(&path);
    path
}

/// Solve 4×4 at `komi_times_two` in segments of `slice`, resuming each from
/// the last snapshot, and return the winner and the number of segments.
fn in_segments(path: &Path, komi_times_two: i32, threads: usize, slice: Duration) -> (Option<Color>, u64) {
    let mut opts = CheckpointOptions {
        write: Some(path.to_path_buf()),
        resume: None,
        every: Some(slice * 3 / 5),
        stop_after: Some(slice),
        build: "test".to_string(),
        at_end: false,
    };
    loop {
        let r = solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(komi_times_two, threads), &opts).expect("checkpointed solve");
        assert!(r.snapshot_error.is_none(), "{:?}", r.snapshot_error);
        if !r.stopped {
            return (r.result.winner, r.segment);
        }
        assert!(r.snapshots >= 1, "a stopped segment writes a last snapshot");
        assert!(r.segment < 2000, "no progress across segments");
        opts.resume = Some(path.to_path_buf());
    }
}

#[test]
fn stopped_and_resumed_solves_agree_with_uninterrupted_ones() {
    for threads in [1, 3] {
        for komi_times_two in [3, 5] {
            let want = solve_dfpn::<4, 4, 1, 16>(&[], cfg(komi_times_two, threads)).winner;
            assert!(want.is_some());
            let path = scratch(&format!("4x4-{komi_times_two}-{threads}.ckpt"));
            let (got, segments) = in_segments(&path, komi_times_two, threads, Duration::from_millis(200));
            assert_eq!(got, want, "komi {komi_times_two}/2, {threads} threads, {segments} segments");
            std::fs::remove_file(&path).ok();
        }
    }
}

#[test]
fn checkpoints_are_refused_by_another_problem_and_kept_from_fresh_runs() {
    let path = scratch("refuse.ckpt");
    let opts = CheckpointOptions {
        write: Some(path.clone()),
        resume: None,
        every: None,
        stop_after: Some(Duration::from_millis(30)),
        build: "test".to_string(),
        at_end: false,
    };
    let r = solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(5, 2), &opts).expect("first segment");
    assert!(path.exists() || !r.stopped);
    if !path.exists() {
        // Solved before the stop: nothing to resume, nothing to refuse.
        return;
    }
    let resume = CheckpointOptions { resume: Some(path.clone()), ..opts.clone() };
    // Other knowledge settings, another table size, another build: refused (another komi seeds, see below).
    for (c, build) in [
        (DfpnConfig { region_interior: 6, ..cfg(5, 2) }, "test"),
        (DfpnConfig { tt_log_buckets: 16, ..cfg(5, 2) }, "test"),
        (cfg(5, 2), "other"),
    ] {
        let o = CheckpointOptions { build: build.to_string(), ..resume.clone() };
        assert!(matches!(solve_dfpn_resumable::<4, 4, 1, 16>(&[], c, &o), Err(CheckpointError::Refused(_))));
    }
    // A fresh run must not overwrite it.
    assert!(matches!(solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(5, 2), &opts), Err(CheckpointError::Refused(_))));
    // The same problem resumes, with more threads.
    let r = solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(5, 4), &CheckpointOptions { stop_after: None, ..resume }).expect("resume");
    assert_eq!(r.segment, 2);
    assert_eq!(r.result.winner, Some(Color::White));
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_checkpoint_seeds_the_runs_of_later_positions() {
    // 4x4 at komi 2.5 is a White win, so White wins after every Black first move.
    let path = scratch("seed.ckpt");
    let base = CheckpointOptions {
        write: Some(path.clone()),
        resume: None,
        every: None,
        stop_after: Some(Duration::from_millis(150)),
        build: "test".to_string(),
        at_end: false,
    };
    let r = solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(5, 2), &base).expect("root segment");
    if !path.exists() {
        // Solved before the stop: there is no checkpoint to seed from.
        assert_eq!(r.result.winner, Some(Color::White));
        return;
    }
    assert!(!r.root_moves.is_empty(), "the root's children are reported");
    let seed = |m: &[Move<4, 4>], write: Option<PathBuf>, stop: Option<Duration>| {
        let o = CheckpointOptions { write, resume: Some(path.clone()), every: None, stop_after: stop, at_end: false, build: "test".to_string() };
        solve_dfpn_resumable::<4, 4, 1, 16>(m, cfg(5, 2), &o)
    };
    for &(code, _, _) in &r.root_moves {
        let m = [Move::<4, 4>::decode(code).expect("move code")];
        let want = solve_dfpn::<4, 4, 1, 16>(&m, cfg(5, 2)).winner;
        let s = seed(&m, None, None).expect("seeded run");
        assert_eq!(s.seeded_from.as_deref(), Some(path.as_path()));
        assert_eq!(s.segment, 1);
        assert_eq!(s.result.winner, want, "after {code}");
        assert_eq!(s.result.winner, Some(Color::White));
    }
    // A child's checkpoint seeds neither its parent nor a sibling.
    if r.root_moves.len() >= 2 {
        let a = [Move::<4, 4>::decode(r.root_moves[0].0).expect("move code")];
        let b = [Move::<4, 4>::decode(r.root_moves[1].0).expect("move code")];
        let child = scratch("seed-child.ckpt");
        let _ = seed(&a, Some(child.clone()), Some(Duration::ZERO)).expect("child segment");
        if child.exists() {
            for m in [&b[..], &[][..]] {
                let o = CheckpointOptions { resume: Some(child.clone()), ..CheckpointOptions { build: "test".to_string(),
        at_end: false, ..CheckpointOptions::default() } };
                assert!(matches!(solve_dfpn_resumable::<4, 4, 1, 16>(m, cfg(5, 2), &o), Err(CheckpointError::Refused(_))));
            }
            std::fs::remove_file(&child).ok();
        }
    }
    std::fs::remove_file(&path).ok();
}

/// A random legal move sequence of up to `max_len` moves (real superko history), as in `dfpn.rs`.
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
fn a_checkpoint_seeds_runs_at_another_komi_soundly() {
    // A short run at komi k1 (any pruning) writes a snapshot that seeds a run at komi k2 (any pruning). The
    // seeded run keeps only the proofs that stay valid at k2, so its verdict must never contradict a fresh
    // unpruned solve at k2, and an unpruned seeded run must reach that verdict.
    let mut rng = Rng::new(0x0C0F_FEE5);
    let sides = [None, Some(Color::Black), Some(Color::White)];
    let mut seeded = 0;
    for sample in 0..60 {
        let Some(prefix) = random_prefix::<4, 4, 1, 16>(&mut rng, 6) else { continue };
        let k1 = 2 * rng.next_below(9) as i32 - 5;
        let k2 = 2 * rng.next_below(9) as i32 - 5;
        let (p1, p2) = (sides[rng.next_below(3) as usize], sides[rng.next_below(3) as usize]);
        if k1 == k2 {
            continue;
        }
        let path = scratch(&format!("komi-{sample}.ckpt"));
        let first = CheckpointOptions {
            write: Some(path.clone()),
            resume: None,
            every: None,
            stop_after: Some(Duration::from_millis(40)),
            build: "test".to_string(),
        at_end: false,
        };
        let r1 = solve_dfpn_resumable::<4, 4, 1, 16>(&prefix, DfpnConfig { prune_for: p1, ..cfg(k1, 2) }, &first);
        r1.expect("first run");
        if !path.exists() {
            continue; // solved before the stop: nothing to seed
        }
        let truth = solve_dfpn::<4, 4, 1, 16>(&prefix, cfg(k2, 2)).winner.expect("unpruned solves decide");
        let o = CheckpointOptions { resume: Some(path.clone()), write: None, stop_after: None, ..first.clone() };
        let got = solve_dfpn_resumable::<4, 4, 1, 16>(&prefix, DfpnConfig { prune_for: p2, ..cfg(k2, 2) }, &o)
            .expect("seeded run");
        assert!(got.komi_seed.is_some(), "seeded at another komi");
        let what = format!("{prefix:?} komi {k1}/2 pruned {p1:?} -> komi {k2}/2 pruned {p2:?}");
        if let Some(w) = got.result.winner {
            assert_eq!(w, truth, "{what}");
        }
        if p2.is_none() {
            assert_eq!(got.result.winner, Some(truth), "{what}");
        }
        seeded += 1;
        std::fs::remove_file(&path).ok();
    }
    assert!(seeded >= 10, "only {seeded} seeded runs");
}
