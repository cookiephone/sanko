# Sanko

[![arXiv](https://img.shields.io/badge/arXiv-2610.29414-b31b1b.svg)](https://arxiv.org/abs/2610.29414)

Sanko solves Go on small rectangular boards under positional superko (PSK), the repetition rule of the Chinese rules, with area scoring and no suicide. It is the solver behind the paper *Solving Go under Positional Superko: Sound Proof Transfer and New Board Values* (Sami Kharma and Florian Schintke, [arXiv:2610.29414](https://arxiv.org/abs/2610.29414), see [Citation](#citation)).

Sanko is a parallel depth-first proof-number search (df-pn) over one shared lock-free table, built on three parts that stay sound when legality depends on the whole game history:

- **Proof transfer.** A stored proof is reused on a new path unless a board that the proof's winner must create has already occurred there. The test runs in time linear in the path length and is exact.
- **Static knowledge.** Life and territory recognition (a variant of Benson's algorithm that holds under PSK play-out) and a solver for small closed regions.
- **Winner-side pruning.** A run may skip the conjectured winner's eye fills and moves inside its own territory. Such a run can only prove a win for that side.

A learned prior (a logistic model over cheap board features) orders the search, and simulation proves the other moves of a node by replaying the proof of one of them, checking every move as the search does. Neither affects results.

## New Board Values

The value of a board is Black's winning margin with optimal play under PSK. It is established by a Black win at komi `v - 0.5` and a White win at komi `v + 0.5`.

| Board | Value | Board | Value | Board | Value | Board | Value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1×10 | 1 | 2×10 | 4 | 3×7 | 21 | 4×5 | 20 |
| 1×11 | 2 | 2×11 | 6 | 3×8 | 24 | 4×6 | 8 |
| 1×12 | 2 | 2×12 | 4\* | 3×9 | 5 | 4×7 | 12 |
| 1×13 | 0\* | 2×13 | 6\* | 3×10 | 8\* | 4×8 | 6\* |
| 1×14 | 1\* | 2×14 | 4\* | 3×11 | 9\* | 5×5 | 25 |
| 1×15 | 2\* | 2×15 | 6\* | | | 5×6 | 4 |
| 1×16 | 1\* | 2×16 | 4\* | | | | |
| 1×17 | 0\* | | | | | | |
| 1×18 | 1\* | | | | | | |
| 1×19 | 2\* | | | | | | |

Every value except that of 5×5 is new under PSK. Earlier solutions of these boards used basic ko (MIGOS, van der Werf and Winands 2009). Values marked with \* had no earlier value under any rules: MIGOS stopped at 1×12 and 2×11, and for 3×10 it found only the lower bound 6. The linear values from 1×9 to 1×15 repeat 0, 1, 2, 2 with period four, but 1×16 is worth 1, not 2, and 1×17, 1×18 and 1×19 return to the pattern with 0, 1 and 2. 4×8, 3×11, 2×16 and 1×19 were solved after the paper, each side on one machine. On 4×8, Black wins at komi 5.5 with D3 in 8.6 hours and White at 6.5 in 6.1 hours (Sanko 0.5.1). On 3×11, Black wins at komi 8.5 with E2 in 49 hours (0.5.1) and White at 9.5 (0.5.3). On 2×16, Black wins at komi 3.5 with H2 in 45 hours and White at 4.5 in 89 hours (0.5.1), so the 2×n values alternate between 4 and 6 from 2×10 to 2×16. On 1×19, Black wins at komi 1.5 with B1 in 54 hours (0.5.3, with ε = 2.0 for the last 6 of them) and White at 2.5 in 46 hours (0.5.5, ε = 1.0). At 32 and 33 points they are the first boards above 30 points with a value under any rules (MIGOS covered boards of up to 30 points). On the boards that both rule sets cover, the PSK value equals the basic-ko value, except on 4×7, which is worth 28 under basic ko but only 12 under PSK. Smaller boards agree with the literature.

[`results/`](results/) holds the output of every run behind these values. The first line of each log names the build (version and a fingerprint of the solver source) and the full command line.

## Build

Requires Rust 1.88 or later.

```sh
cargo build --release
cargo test --workspace --release
```

The binary is `target/release/sanko`, built for the host CPU (`.cargo/config.toml` sets `target-cpu=native`). A profile-guided build is about 1.5 times faster and runs the same search. Train an instrumented binary on a few short solves, then build with the merged profile:

```sh
rustup component add llvm-tools
RUSTFLAGS="-C target-cpu=native -Cprofile-generate=$PWD/pgo" cargo build --release --target-dir target-gen
F="--regions --region-interior 8 --keep-tempo --threads 4 --pn-tt-log 24 --proof-log 24 --sc-log 23 --node-limit 15000000"
for a in "--rows 5 --cols 5 --komi 24.5 --moves C3 --prune-for black" "--rows 4 --cols 6 --komi 7.5 --prune-for black" \
         "--rows 4 --cols 6 --komi 8.5 --prune-for white" "--rows 4 --cols 7 --komi 23.5 --moves D3 --prune-for white" \
         "--rows 3 --cols 10 --komi 6.5 --moves E2 --prune-for white" "--rows 1 --cols 13 --komi 0.5"
do
  target-gen/release/sanko $a $F > /dev/null
done
"$(rustc --print sysroot)"/lib/rustlib/*/bin/llvm-profdata merge -o pgo/merged.profdata pgo/*.profraw
RUSTFLAGS="-C target-cpu=native -Cprofile-use=$PWD/pgo/merged.profdata" cargo build --release
```

The solver is also published on crates.io as [`sanko`](https://crates.io/crates/sanko), with the library crates [`sanko-core`](https://crates.io/crates/sanko-core) and [`sanko-search`](https://crates.io/crates/sanko-search). `cargo install sanko` applies neither `.cargo/config.toml` nor the workspace release profile, which makes it up to about 10% slower than `cargo build --release` here. This gives the same build as that:

```sh
RUSTFLAGS="-C target-cpu=native" CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_PANIC=abort cargo install sanko
```

## Usage

```sh
sanko --rows 4 --cols 4 --komi 1.5 --prune-for black --regions --threads 4
sanko --rows 5 --cols 5 --komi 24.5 --moves C3 --prune-for black --regions --region-interior 8 \
      --keep-tempo --threads 192 --pn-tt-log 30 --proof-log 31 --sc-log 30
```

On a terminal, Sanko shows a live panel and board diagrams. With `--plain`, or when the output is not a terminal, it prints a stable plain format: the build and command line, then the proven winner (`winner: Some(Black)`, `Some(White)` or `None` for unknown), the expansion count, the wall time and, when the side to move wins, its winning move. `sanko --help` and `sanko <command> --help` describe every option.

| Option | Meaning |
| --- | --- |
| `--rows R --cols C` | board size, at most 49 points |
| `--komi K` | komi added to White's area (a draw counts as a White win) |
| `--moves C3,D4,pass` | moves played from the empty board first (Go coordinates, A1 bottom left, no I) |
| `--prune-for black\|white` | prove a win for this side only (any other outcome is reported as unknown) |
| `--keep-tempo` | with pruning: keep one skipped move per node |
| `--regions`, `--region-interior N` | closed-region solver, with interior moves in regions of at most N points |
| `--threads N` | worker threads sharing one table |
| `--vl-free K` | with pruning: at the pruned side's nodes, the first K other threads inside a child do not count for its virtual loss (default an eighth of the threads, 0 up to version 0.5.4) |
| `--pn-tt-log L --proof-log P --sc-log S` | tables of 64 B·2^L, 192 B·2^P (256 B above 32 points) and 8 B·2^S |
| `--proof-gib G` | proof table of G GiB, any size |
| `--node-limit N` | stop after N expansions (result unknown) |
| `--prior learned\|eye` | initial proof numbers from the learned prior (default) or from White's eye space |
| `--eps E` | growth factor of the 1+ε rule (default 0.15, or 2.0 on boards of one row from version 0.5.6, and 0.3 up to version 0.5.2) |
| `--simulate N` | simulation of sibling proofs, at most N nodes per attempt (default 100, 0 turns it off) |
| `--checkpoint PATH`, `--checkpoint-every D` | write snapshots of the search to PATH every D (default 6h) and at `--stop-after` |
| `--resume PATH` | continue from a snapshot (same build, problem and table sizes), or seed a run of another position of the same game (any position from version 0.5.7, a later one before) or at another komi with it |
| `--snapshot-at-end` | with `--checkpoint`: also write a snapshot when the search ends with a result |
| `--snapshot-file-gib G` | with `--checkpoint`: write each snapshot as files of at most G GiB (`PATH`, `PATH.1`, ...) for file systems that refuse larger files, found again from the snapshot's header (from version 0.5.7) |
| `--stop-after D` | stop after D of wall time (such as 9d23h), with a last snapshot when writing them (exit code 75) |

The reported runs use `--regions --region-interior 8 --keep-tempo` with the defaults otherwise, which included `--eps 0.3` up to version 0.5.2. Pass `--eps 0.3` to repeat their exact expansion counts with a later version. The winners do not depend on it. From version 0.5.5, pruned parallel runs let up to an eighth of the threads work in the same child at the pruned side's nodes (`--vl-free`), which took 49% to 90% of the time of 0.5.4 on six pruned problems with 192 threads. `--vl-free 0` restores the earlier selection. From version 0.5.6, boards of one row default to `--eps 2.0`: their long games with near-equal moves made 0.15 search the same lines hundreds of times, and 2.0 solved 1×18 Black at komi 0.5 in 10 minutes against more than 2 hours with 0.15 (128 threads) and 1×17 White at 0.5 in 2.6 h against 17.4 h for the earlier reported run, while boards with more rows are slower with it. The paper's runs were made with versions 0.4.1 and 0.4.2, before simulation, so later versions find the same values with fewer expansions. Boards of up to 25 points use `--pn-tt-log 30 --proof-log 31 --sc-log 30` (about 0.45 TiB), larger boards `--pn-tt-log 31 --proof-log 31 --sc-log 29`. 4×8, 3×11 and 2×16 were solved with 0.5.1 and 0.5.3 and `--pn-tt-log 31 --proof-log 32 --sc-log 32` (about 0.9 TiB), as was Black's side of 1×19, and White's side of 1×19 with 0.5.5, `--pn-tt-log 30 --proof-log 31 --sc-log 30` and 128 threads. On multi-socket machines, run under `numactl --interleave=all`. Parallel runs are not deterministic, so expansion counts vary between runs while the winner does not. Serial runs are deterministic.

Other commands:

- `sanko selftest` solves random positions (random legal games, so real superko histories, and random komi) with the full solver and with an oracle that reuses results only on the identical game. `--oracle-no-static` also removes all static knowledge from the oracle.
- `sanko staticcheck` sets the komi exactly at the point where a static rule declares a winner and checks the verdict against search.
- `sanko pskcheck --moves ...` replays a game and lists the moves that superko forbids but basic ko would allow.
- `sanko moves --moves ...` lists a position's moves as the search generates them (symmetric moves merged, the pruned side's skipped moves left out), with their proof and disproof numbers.
- `sanko checkpoint info PATH` prints a checkpoint's header: build, problem, table sizes, and the expansions and time so far.
- `sanko bench` solves seeded random positions near a prefix, for comparing options.
- `sanko train` fits the learned prior.

## Long Runs

A solve can run as a series of segments that each fit a job's time limit. df-pn keeps its whole state in the tables, so a snapshot of the tables is enough to continue. `--checkpoint PATH` writes one every `--checkpoint-every` (default 6h) and at `--stop-after`, and `--resume PATH` continues from it. Writing a snapshot pauses the search and does not change it.

```sh
S="sanko --rows 6 --cols 6 --komi 3.5 --moves C3,D4 --prune-for black --regions --region-interior 8 --keep-tempo \
   --threads 192 --pn-tt-log 31 --proof-log 32 --sc-log 32"
$S --checkpoint run.ckpt --stop-after 9d23h                       # the first segment
$S --resume run.ckpt --checkpoint run.ckpt --stop-after 9d23h     # every later one, until the exit code is 0
```

A segment exits with code 75 when it stopped with a snapshot written, and with 0 once the search has ended. A checkpoint is resumed only by the same build (source fingerprint), for the same position, komi, pruning and knowledge options, with the same table sizes. The thread count, `--eps`, the prior and `--simulate` may change between segments. `--checkpoint` refuses an existing file unless it is also the `--resume` file.

A snapshot can also seed a different run. With other `--moves`, a run of another position of the same game (a later one, or a sibling's) starts with every proof of the snapshot, each reused only where it passes the transfer test on the new path. With another `--komi`, the run keeps the proofs that stay valid there: Black's wins at a lower komi, White's at a higher one, each only if the earlier run was not pruned for the other side. A value search can chain its runs this way, each komi resuming from the previous one's snapshot (`--snapshot-at-end` writes one when a run ends with its result). A run with checkpoints ends by listing the root's moves with their numbers (`root moves:`), the moves that a split of the position into parts needs.

A snapshot is as large as the tables (about 1.1 TiB at the sizes above), and while it replaces the previous one both exist, so leave room for twice that. The static cache is not saved. A resumed search restarts from the root, so its expansion counts differ slightly from those of an uninterrupted run, while the result does not.

## Learned Prior

The prior gives each new node the initial numbers `1 + S·(-ln p)`, where `p` is the modelled probability that the side to move wins. To fit new weights, dump a sample of the proofs of a few solves and train:

```sh
sanko --rows 3 --cols 7 --komi 20.5 --prune-for black --regions --dump-proofs 37b.bin --dump-rate 50
sanko --rows 2 --cols 10 --komi 4.5 --prune-for white --regions --dump-proofs 210w.bin --dump-rate 50
sanko train --validate 37b.bin 210w.bin        # prints --prior-weights W1,...,W14
```

`--prior-weights` then replaces the built-in weights in any solve.

## The 4×7 Proof Chain

MIGOS's published optimal line for 4×7 (B+28 under basic ko) fails under PSK. These small proofs locate the failure and take seconds. Black results come from Black-pruned runs and White results from White-pruned runs, so each is trusted for its winner.

```sh
S="sanko --rows 4 --cols 7 --threads 4 --regions --region-interior 8 --keep-tempo --pn-tt-log 24 --proof-log 24 --sc-log 24"
L24=D3,C2,C3,E2,D2,E3,F1,E1,F3,F2,E4,F4,G4,D4,C4,G3,G2,G1,F4,G3,D4,B2,B3,D1   # MIGOS's line, moves 1-24
L25=$L24,A3
sanko pskcheck --rows 4 --cols 7 --moves $L25,A2,C1,B1,G2,A4,F1     # the whole line is legal under PSK
$S --komi 27.5 --prune-for black --moves $L25,A2,C1,B1,G2,A4,F1     # its end is a Black win
$S --komi 27.5 --prune-for black --moves $L25,A2                    # after move 26: Black wins (C1)
$S --komi 27.5 --prune-for black --moves $L24                       # after move 24: unknown
$S --komi 27.5 --prune-for white --moves $L25                       # after move 25: White wins with G2
$S --komi 2.5 --prune-for black --moves $L25,G2                     # after G2: Black wins at 2.5 (A2)
$S --komi 3.5 --prune-for white --moves $L25,G2                     # and White at 3.5, so G2 is worth +3
sanko pskcheck --rows 4 --cols 7 --moves $L25,G2,A2,C1,A1,B4,A4,pass,pass   # the optimal seki line
$S --komi 27.5 --prune-for white --moves $L25,G2,B1,C1,A2,A1        # moonshine life: White wins
$S --komi 27.5 --prune-for black --moves $L25,G2,B1,C1,A2,pass,F1   # if Black could retake, Black kills all
```

## Citation

If you use Sanko or its results, please cite the paper:

```bibtex
@misc{kharma2026solvingpositionalsuperkosound,
  title={Solving Go under Positional Superko: Sound Proof Transfer and New Board Values},
  author={Sami Kharma and Florian Schintke},
  year={2026},
  eprint={2610.29414},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2610.29414},
}
```

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
