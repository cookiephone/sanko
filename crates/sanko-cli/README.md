# Sanko

[![arXiv](https://img.shields.io/badge/arXiv-2610.29414-b31b1b.svg)](https://arxiv.org/abs/2610.29414)

Sanko solves Go on small rectangular boards under positional superko (PSK), the repetition rule of the Chinese rules, with area scoring and no suicide. It is the solver behind the paper *Solving Go under Positional Superko: Sound Proof Transfer and New Board Values* (Sami Kharma and Florian Schintke, [arXiv:2610.29414](https://arxiv.org/abs/2610.29414), see [Citation](#citation)). It established the PSK values of 25 boards, 13 of which had no earlier value under any rules, among them 1×18, 2×15, 3×10, 4×8 and 3×11, the first boards above 30 points with a value.

This crate is the command-line solver. The search is in [`sanko-search`](https://crates.io/crates/sanko-search) and the rules engine in [`sanko-core`](https://crates.io/crates/sanko-core). The source, the logs of every run behind the board values and the full documentation are at [github.com/cookiephone/sanko](https://github.com/cookiephone/sanko).

## Install

Requires Rust 1.88 or later.

```sh
cargo install sanko
```

This builds for baseline x86-64 with cargo's default release profile, up to about 10% slower than the build of the repository. That build targets the host CPU and uses the repository's release profile, which `cargo install` does not apply. To get it:

```sh
RUSTFLAGS="-C target-cpu=native" CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_PANIC=abort cargo install sanko
```

A profile-guided build is about 1.5 times faster still and needs a checkout of the repository, see its README.

`sanko --version` prints the version and `(source crates.io)`. A build from the repository prints a fingerprint of the solver source instead.

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
| `--eps E` | growth factor of the 1+ε rule (default 0.15, 0.3 up to version 0.5.2) |
| `--simulate N` | simulation of sibling proofs, at most N nodes per attempt (default 100, 0 turns it off) |
| `--checkpoint PATH`, `--checkpoint-every D` | write snapshots of the search to PATH every D (default 6h) and at `--stop-after` |
| `--resume PATH` | continue from a snapshot (same build, problem and table sizes), or seed a run of a later position or at another komi with it |
| `--snapshot-at-end` | with `--checkpoint`: also write a snapshot when the search ends with a result |
| `--stop-after D` | stop after D of wall time (such as 9d23h), with a last snapshot when writing them (exit code 75) |

The repository README gives the settings of the reported runs and the table sizes they used, and explains long runs in segments with checkpoints.

Other commands:

- `sanko selftest` solves random positions (random legal games, so real superko histories, and random komi) with the full solver and with an oracle that reuses results only on the identical game. `--oracle-no-static` also removes all static knowledge from the oracle.
- `sanko staticcheck` sets the komi exactly at the point where a static rule declares a winner and checks the verdict against search.
- `sanko pskcheck --moves ...` replays a game and lists the moves that superko forbids but basic ko would allow.
- `sanko moves --moves ...` lists a position's moves as the search generates them (symmetric moves merged, the pruned side's skipped moves left out), with their proof and disproof numbers.
- `sanko checkpoint info PATH` prints a checkpoint's header: build, problem, table sizes, and the expansions and time so far.
- `sanko bench` solves seeded random positions near a prefix, for comparing options.
- `sanko train` fits the learned prior.

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

Apache License 2.0, see [`LICENSE`](https://github.com/cookiephone/sanko/blob/HEAD/LICENSE).
