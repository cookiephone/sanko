//! Checkpoints: a solve that is stopped and resumed many times must reach the
//! same winner as an uninterrupted solve, serially and in parallel, and a
//! checkpoint must be refused by a different problem and must not be
//! overwritten by a fresh run.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sanko_core::Color;
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
    };
    let r = solve_dfpn_resumable::<4, 4, 1, 16>(&[], cfg(5, 2), &opts).expect("first segment");
    assert!(path.exists() || !r.stopped);
    if !path.exists() {
        // Solved before the stop: nothing to resume, nothing to refuse.
        return;
    }
    let resume = CheckpointOptions { resume: Some(path.clone()), ..opts.clone() };
    // Another komi, another table size, another build: refused.
    for (c, build) in [(cfg(3, 2), "test"), (DfpnConfig { tt_log_buckets: 16, ..cfg(5, 2) }, "test"), (cfg(5, 2), "other")] {
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
