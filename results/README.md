# Results

The output of every solver run behind the values, speed measurements and ablations reported in the paper.

- [`runs.md`](runs.md): one row per solve (board, komi, opening moves, pruned side, winner, winning move at the root, expansions, wall time, threads), followed by the full command of each solve. Generated from the logs by `campaign_status.py`, which also checks every winner against the reported board value.
- [`logs/`](logs/): the solver's output of each solve. The first line names the build (version and a fingerprint of the solver source) and the full command line.
- [`suites/`](suites/): the benchmark suites behind the ablations and the prior comparison (`sanko bench` output, one file per configuration and suite).
- [`validation/`](validation/): the selftests, threshold probes and large-board selftests.

## Build and Hardware

Every run used Sanko 0.4.1, source fingerprint `63014da88da067d1`, as the first line of each log shows. The binary is a profile-guided build for the AMD EPYC 9475F, which runs the same search as the plain build (`cargo build --release` prints the same fingerprint). Every solve ran on one machine with two AMD EPYC 9475F CPUs (192 threads, 1.5 TB). Parallel runs are not deterministic: expansion counts vary between runs, while the winner does not. Serial runs are deterministic.

## How to Read a Run

- **Komi** is added to White's area. The PSK value `v` of a board is established by a Black win at komi `v - 0.5` and a White win at komi `v + 0.5`. When `v` equals the number of points, the White side holds trivially.
- **Pruned runs** (`--prune-for black|white`) skip the pruned side's own eye fills and moves inside its own territory, except for one kept tempo move per node. They can only prove a win for that side. Any other outcome is reported as unknown (`winner: None`). Unpruned runs are trusted in both directions.
- **Moves** in commands are Go coordinates or `rc` digits (row 0 at the top), or `pass`. The table shows Go coordinates (A1 at the bottom left, no I).
- **Split proofs.** The upper bounds of 3×10 and 5×6 are proved with one White-pruned run per symmetry-unique first move of Black.
- **Times** are wall-clock and include setting up the tables, which takes about 45 seconds and dominates the time of the small boards.
