# sanko-search

The search of [Sanko](https://github.com/cookiephone/sanko), a solver for Go on small rectangular boards under positional superko (PSK): a parallel depth-first proof-number search (df-pn) over one shared lock-free table.

- **Proof transfer.** A stored proof is reused on a new path unless a board that the proof's winner must create has already occurred there. The test runs in time linear in the path length and is exact.
- **Static knowledge.** Life and territory recognition from [`sanko-core`](https://crates.io/crates/sanko-core) and a solver for small closed regions, with a shared cache.
- **Winner-side pruning.** A run may skip the conjectured winner's eye fills and moves inside its own territory. Such a run can only prove a win for that side.
- **Ordering.** A learned prior gives the initial proof numbers, and simulation proves the other moves of a node by replaying the proof of one of them, checking every move as the search does. Neither affects results.

`solve_dfpn` solves the position after a move prefix under a `DfpnConfig` and returns the proven winner, or none, with statistics.

The API serves the [`sanko`](https://crates.io/crates/sanko) solver and is not stable: it may change in any 0.x release.

## License

Apache License 2.0, see [`LICENSE`](https://github.com/cookiephone/sanko/blob/HEAD/LICENSE).
