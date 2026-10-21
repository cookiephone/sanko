# sanko-core

The board representation and rules engine of [Sanko](https://github.com/cookiephone/sanko), a solver for Go on small rectangular boards. It implements the Chinese rules as written, with area scoring, no suicide and positional superko (PSK), played out to the end.

- Bitboards of `ROWS × COLS` points, at most 49, with the size as const generics. The macros `define_board!` and `define_position!` and aliases such as `Board5x5` and `Position5x5` bind the parameters consistently.
- Positions with the full board history that superko needs: move legality, play and unmake.
- 128-bit Zobrist keys that are exact (distinct boards have distinct keys) on every board of up to 49 points.
- Static knowledge that stays sound under PSK play-out: unconditional life by a variant of Benson's algorithm, and territory recognition.
- Area scoring and the board symmetries.

The API serves the [`sanko`](https://crates.io/crates/sanko) solver and is not stable: it may change in any 0.x release.

## License

Apache License 2.0, see [`LICENSE`](https://github.com/cookiephone/sanko/blob/HEAD/LICENSE).
