# Sanko

[![Paper](https://img.shields.io/badge/paper-arXiv%20preprint%20coming%20soon-b31b1b.svg)](#citation)

Sanko solves Go on small rectangular boards under positional superko (PSK), the repetition rule of the Chinese rules, with area scoring and no suicide. It is the solver behind the paper *Solving Go under Positional Superko: Sound Proof Transfer and New Board Values* (Sami Kharma and Florian Schintke, preprint to appear on arXiv, see [Citation](#citation)).

Sanko is a parallel depth-first proof-number search (df-pn) over one shared lock-free table, built on three parts that stay sound when legality depends on the whole game history:

- **Proof transfer.** A stored proof is reused on a new path unless a board that the proof's winner must create has already occurred there. The test runs in time linear in the path length and is exact.
- **Static knowledge.** Life and territory recognition (a variant of Benson's algorithm that holds under PSK play-out) and a solver for small closed regions.
- **Winner-side pruning.** A run may skip the conjectured winner's eye fills and moves inside its own territory. Such a run can only prove a win for that side.

A learned prior (a logistic model over cheap board features) orders the search. It does not affect results.

## New Board Values

The value of a board is Black's winning margin with optimal play under PSK. It is established by a Black win at komi `v - 0.5` and a White win at komi `v + 0.5`.

| Board | Value | Board | Value | Board | Value | Board | Value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1×10 | 1 | 2×10 | 4 | 3×7 | 21 | 4×5 | 20 |
| 1×11 | 2 | 2×11 | 6 | 3×8 | 24 | 4×6 | 8 |
| 1×12 | 2 | 2×12 | 4\* | 3×9 | 5 | 4×7 | 12 |
| 1×13 | 0\* | 2×13 | 6\* | 3×10 | 8\* | 5×5 | 25 |
| 1×14 | 1\* | 2×14 | 4\* | | | 5×6 | 4 |
| 1×15 | 2\* | | | | | | |

Every value except that of 5×5 is new under PSK. Earlier solutions of these boards used basic ko (MIGOS, van der Werf and Winands 2009). Values marked with \* had no earlier value under any rules: MIGOS stopped at 1×12 and 2×11, and for 3×10 it found only the lower bound 6. On the boards that both rule sets cover, the PSK value equals the basic-ko value, except on 4×7, which is worth 28 under basic ko but only 12 under PSK. Smaller boards agree with the literature.

[`results/`](results/) holds the output of every run behind these values. The first line of each log names the build (version and a fingerprint of the solver source) and the full command line.

## Build

Requires Rust 1.87 or later.

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

## Usage

```sh
sanko --rows 4 --cols 4 --komi 1.5 --prune-for black --regions --threads 4
sanko --rows 5 --cols 5 --komi 24.5 --moves C3 --prune-for black --regions --region-interior 8 \
      --keep-tempo --threads 192 --pn-tt-log 30 --proof-log 31 --sc-log 30
```

On a terminal, Sanko shows a live panel and board diagrams. With `--plain`, or when the output is not a terminal, it prints a stable plain format: the build and command line, then the proven winner (`winner: Some(Black)`, `Some(White)` or `None` for unknown), the expansion count, the wall time and, when the side to move wins, its winning move. `sanko --help` and `sanko <command> --help` describe every option.

| Option | Meaning |
| --- | --- |
| `--rows R --cols C` | board size, at most 30 points |
| `--komi K` | komi added to White's area (a draw counts as a White win) |
| `--moves C3,D4,pass` | moves played from the empty board first (Go coordinates, A1 bottom left, no I) |
| `--prune-for black\|white` | prove a win for this side only (any other outcome is reported as unknown) |
| `--keep-tempo` | with pruning: keep one skipped move per node |
| `--regions`, `--region-interior N` | closed-region solver, with interior moves in regions of at most N points |
| `--threads N` | worker threads sharing one table |
| `--pn-tt-log L --proof-log P --sc-log S` | tables of 64 B·2^L, 192 B·2^P (256 B above 32 points) and 8 B·2^S |
| `--proof-gib G` | proof table of G GiB, any size |
| `--node-limit N` | stop after N expansions (result unknown) |
| `--prior learned\|eye` | initial proof numbers from the learned prior (default) or from White's eye space |
| `--eps E` | growth factor of the 1+ε rule (default 0.3) |

The reported runs use `--regions --region-interior 8 --keep-tempo` with the defaults otherwise. Boards of up to 25 points use `--pn-tt-log 30 --proof-log 31 --sc-log 30` (about 0.45 TiB), larger boards `--pn-tt-log 31 --proof-log 31 --sc-log 29`. On multi-socket machines, run under `numactl --interleave=all`. Parallel runs are not deterministic, so expansion counts vary between runs while the winner does not. Serial runs are deterministic.

Other commands:

- `sanko selftest` solves random positions (random legal games, so real superko histories, and random komi) with the full solver and with an oracle that reuses results only on the identical game. `--oracle-no-static` also removes all static knowledge from the oracle.
- `sanko staticcheck` sets the komi exactly at the point where a static rule declares a winner and checks the verdict against search.
- `sanko pskcheck --moves ...` replays a game and lists the moves that superko forbids but basic ko would allow.
- `sanko bench` solves seeded random positions near a prefix, for comparing options.
- `sanko train` fits the learned prior.

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

The paper will appear on arXiv shortly. This section will then be updated with its link and a BibTeX entry.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
