//! Command-line interface of the Sanko Go solver.
//!
//! `sanko --rows R --cols C --komi K` solves a position, and subcommands run the
//! validation and benchmark modes. On a terminal the output is formatted for
//! reading (see [`ui`]). Otherwise, or with `--plain`, it is plain text in a
//! stable format for logs and scripts.

mod panel;
mod ui;

use std::fmt::Write as _;
use std::io::IsTerminal;
use std::process::ExitCode;
use std::time::Instant;

use clap::error::ErrorKind;
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use sanko_core::{Color, HashHistory, Move, PlayError, Point, Position, Rng};
use sanko_search::{DfpnConfig, DfpnResult, solve_dfpn};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const SOURCE: &str = env!("SANKO_SOURCE");
const LONG_VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (source ", env!("SANKO_SOURCE"), ")");

const LONG_ABOUT: &str = "\
Sanko proves who wins a Go position under the Chinese rules as written: area scoring, no
suicide, no move may recreate an earlier whole-board position (positional superko), two
passes end the game, and komi is added to White. The game is played out instead of agreeing
on dead stones. It runs depth-first proof-number search with sound proof transfer between
paths and static life-and-death knowledge.

Without a subcommand, sanko solves the position given by --rows, --cols, --komi and --moves.";

const EXAMPLES: &str = "\
Examples:
  sanko --rows 3 --cols 3 --komi 8.5
  sanko --rows 4 --cols 4 --komi 1.5 --prune-for black --regions --eye-init 8 --eye-area \\
        --eye-weight 4 --eye-komi --region-interior 8 --eps 0.15 --keep-tempo
  sanko --rows 5 --cols 5 --komi 24.5 --moves C3,B4 --threads 8 --pn-tt-log 26
  sanko selftest --rows 3 --cols 4 --positions 400 --regions --oracle-no-static
  sanko pskcheck --rows 4 --cols 7 --moves D3,C2,C3,E2";

#[derive(Parser, Debug)]
#[command(
    name = "sanko",
    bin_name = "sanko",
    version = LONG_VERSION,
    about = "Solve small-board Go under the Chinese rules (positional superko).",
    long_about = LONG_ABOUT,
    after_help = EXAMPLES,
    override_usage = "sanko --rows <R> --cols <C> [OPTIONS]
       sanko <COMMAND> [OPTIONS]",
    max_term_width = 100,
    args_override_self = true,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    board: TopBoardArgs,
    #[command(flatten)]
    solve: SolveOptions,
}

/// The board of the default command (solve), required unless a subcommand is given.
#[derive(Args, Debug, Clone, Copy)]
#[command(next_help_heading = "Board")]
struct TopBoardArgs {
    /// Number of rows.
    #[arg(long, value_name = "R")]
    rows: Option<usize>,
    /// Number of columns (the board may have at most 30 points).
    #[arg(long, value_name = "C")]
    cols: Option<usize>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Solve a position (the default when no subcommand is given).
    Solve(SolveArgs),
    /// Soundness test: random positions solved by the full solver and by an exact oracle.
    #[command(long_about = "\
Differential soundness test. Each random position (random legal moves from the empty board,
so with a real superko history, and a random komi) is solved twice: by the solver with the
given options, and by an oracle that reuses proofs only on the identical superko game (and,
with --oracle-no-static, uses no static knowledge). Any disagreement is reported.")]
    Selftest(SelftestArgs),
    /// Stress test of the static rules at the exact komi where they decide.
    #[command(long_about = "\
Stress test of the static rules. For random positions where a side has secure points,
the komi is set so that the static rule barely declares that side the winner, and the solver
(with the closed-region solver on) is compared with an oracle that has no static knowledge.
A third probe sets the kill-all komi for Black. Any disagreement is a counterexample.")]
    Staticcheck(StaticcheckArgs),
    /// Replay moves and show where superko forbids a move that basic ko would allow.
    Pskcheck(PskcheckArgs),
    /// Benchmark: solve random positions near a prefix (compare solver options).
    #[command(long_about = "\
Benchmark suite: --bench-n positions reached from --moves by --bench-plies random legal
stone moves (seeded by --rng-seed), each solved with the given options and a node limit.
Serial runs are deterministic, so two configurations can be compared problem by problem.")]
    Bench(BenchArgs),
    /// Fit the learned prior to proof dumps (from --dump-proofs) and print its weights.
    #[command(long_about = "\
Fit the learned prior to proof dumps written by solves with --dump-proofs. The model is a
logistic regression of the winner on the prior's board features, fitted by Newton's method
with every dump weighted equally. The weights are printed as a --prior-weights argument.
With --validate, each dump is also predicted by a model fitted to the other dumps.")]
    Train(TrainArgs),
}

#[derive(Args, Debug)]
struct TrainArgs {
    /// Proof dumps written by --dump-proofs.
    #[arg(required = true, value_name = "DUMP")]
    dumps: Vec<std::path::PathBuf>,
    /// Also predict each dump with a model fitted to the others.
    #[arg(long)]
    validate: bool,
    /// L2 regularisation strength (> 0).
    #[arg(long, value_name = "L", default_value_t = 1e-4, value_parser = parse_l2)]
    l2: f64,
}

#[derive(Args, Debug, Clone, Copy)]
#[command(next_help_heading = "Board")]
struct BoardArgs {
    /// Number of rows.
    #[arg(long, value_name = "R")]
    rows: usize,
    /// Number of columns (the board may have at most 30 points).
    #[arg(long, value_name = "C")]
    cols: usize,
}

#[derive(Args, Debug, Clone)]
#[command(next_help_heading = "Position")]
struct PositionArgs {
    /// Komi added to White's area. A half-integer avoids draws. A draw counts as a White win.
    #[arg(long, value_name = "K", default_value_t = 0.5, allow_negative_numbers = true, value_parser = parse_komi)]
    komi: f64,
    /// Moves played from the empty board first, comma-separated: Go coordinates (C3, with A1
    /// bottom left and no column I), 0-based row-column digits counted from the top (22), or pass.
    #[arg(long, value_name = "MOVES", default_value = "", hide_default_value = true)]
    moves: String,
}

/// Solver options shared by every mode that searches.
#[derive(Args, Debug, Clone, Copy)]
struct SolverArgs {
    /// Prove a win for this side only: its own eye fills and moves inside its own territory
    /// are skipped, so a win for the other side is reported as unknown.
    #[arg(long, value_enum, value_name = "SIDE", help_heading = "Pruning")]
    prune_for: Option<Side>,
    /// With --prune-for: keep one skipped move per node (a tempo move that changes the board).
    #[arg(long, help_heading = "Pruning")]
    keep_tempo: bool,
    /// Use the closed-region solver (Black kills everything inside enclosed regions).
    #[arg(long, overrides_with = "no_regions", help_heading = "Knowledge")]
    regions: bool,
    /// Turn the closed-region solver off (overrides an earlier --regions).
    #[arg(long, overrides_with = "regions", help_heading = "Knowledge")]
    no_regions: bool,
    /// Closed-region solver: Black may also play interior points in regions of at most N points.
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Knowledge")]
    region_interior: u32,
    /// Kill-all prior: initial proof numbers from White's eye space, threshold T (0 = off).
    #[arg(long, value_name = "T", default_value_t = 0, help_heading = "Knowledge")]
    eye_init: u32,
    /// Eye prior: White stones without a Black neighbour count as eye space too.
    #[arg(long, help_heading = "Knowledge")]
    eye_area: bool,
    /// Eye prior: extra weight per existing White eye point.
    #[arg(long, value_name = "W", default_value_t = 0, help_heading = "Knowledge")]
    eye_weight: u32,
    /// Eye prior: count only the area White needs beyond what Black can concede at this komi.
    #[arg(long, help_heading = "Knowledge")]
    eye_komi: bool,
    /// Growth factor of the 1+ε rule for child thresholds (0 = plain df-pn).
    #[arg(long, value_name = "E", default_value_t = 0.3, help_heading = "Search")]
    eps: f64,
    /// Worker threads sharing one table.
    #[arg(long, value_name = "N", default_value_t = 1, help_heading = "Search")]
    threads: usize,
    /// Ablation: reuse proofs only on the identical superko game (no proof transfer).
    #[arg(long, help_heading = "Search")]
    exact_only: bool,
    /// Ablation: parallel search without virtual loss.
    #[arg(long, help_heading = "Search")]
    no_virtual_loss: bool,
    /// Initial proof and disproof numbers of new nodes: the learned prior or the eye-space prior.
    #[arg(long, value_enum, default_value_t = PriorKind::Learned, help_heading = "Knowledge")]
    prior: PriorKind,
    /// Learned prior: comma-separated weights replacing the built-in ones (as printed by `sanko train`).
    #[arg(long, value_name = "W", value_parser = parse_prior_weights, help_heading = "Knowledge")]
    prior_weights: Option<PriorWeights>,
    /// Learned prior: scale of the initial numbers, 1 + S·(−ln p) for win probability p.
    #[arg(long, value_name = "S", default_value_t = sanko_search::prior::DEFAULT_PRIOR_SCALE, value_parser = parse_scale, help_heading = "Knowledge")]
    prior_scale: f32,
    /// Ablation: classify children without prefetching their table lines first.
    #[arg(long, help_heading = "Search")]
    no_prefetch: bool,
    /// Not our rules: MIGOS's end-of-game convention (one-liberty chains are dead if the
    /// opponent has none), for rule comparisons only.
    #[arg(long, help_heading = "Rules")]
    atari_dead: bool,
}

#[derive(Args, Debug, Clone, Copy)]
#[command(next_help_heading = "Tables")]
#[allow(clippy::struct_field_names)] // the fields mirror the flag names
struct TableArgs {
    /// Unproven-entry table: 2^L buckets of 64 bytes (L at most 34).
    #[arg(long, value_name = "L", default_value_t = 22, value_parser = clap::value_parser!(u32).range(0..=34))]
    pn_tt_log: u32,
    /// Proof table: 2^P buckets of 192 bytes (256 above 32 points, P at most 34). Default: L - 1.
    #[arg(long, value_name = "P", value_parser = clap::value_parser!(u32).range(0..=34))]
    proof_log: Option<u32>,
    /// Proof table size in GiB, in place of --proof-log (any size, not only a power of two).
    #[arg(long, value_name = "G", conflicts_with = "proof_log", value_parser = parse_gib)]
    proof_gib: Option<f64>,
    /// Static-safety cache: 2^S slots of 8 bytes.
    #[arg(long, value_name = "S", default_value_t = 22)]
    sc_log: u32,
}

#[derive(Args, Debug, Clone, Copy)]
#[command(next_help_heading = "Output")]
struct OutputArgs {
    /// Plain text with a stable format (the default when output is not a terminal).
    #[arg(long)]
    plain: bool,
    /// When to use colours.
    #[arg(long, value_enum, value_name = "WHEN", default_value_t = ColorWhen::Auto)]
    color: ColorWhen,
}

#[derive(Args, Debug, Clone)]
struct SolveArgs {
    #[command(flatten)]
    board: BoardArgs,
    #[command(flatten)]
    options: SolveOptions,
}

#[derive(Args, Debug, Clone)]
struct SolveOptions {
    #[command(flatten)]
    position: PositionArgs,
    #[command(flatten)]
    solver: SolverArgs,
    #[command(flatten)]
    tables: TableArgs,
    /// Stop after N expansions and report the result as unknown (0 = no limit).
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Search")]
    node_limit: u64,
    /// Progress lines on stderr (on a terminal a live panel is shown instead).
    #[arg(long, help_heading = "Output")]
    progress: bool,
    /// Write a sample of proofs (features, winner, work) to FILE, for training the learned prior.
    #[arg(long, value_name = "FILE", help_heading = "Output")]
    dump_proofs: Option<String>,
    /// With --dump-proofs: keep about one proof in N.
    #[arg(long, value_name = "N", default_value_t = 100, help_heading = "Output")]
    dump_rate: u64,
    #[command(flatten)]
    output: OutputArgs,
}

/// Where the initial proof and disproof numbers of new nodes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum PriorKind {
    /// A logistic model over cheap board features (built-in weights, or --prior-weights).
    Learned,
    /// White's black-free area (--eye-init, --eye-area, --eye-weight, --eye-komi).
    Eye,
}

/// Weights of the learned prior (see `--prior-weights`).
#[derive(Clone, Copy, Debug)]
struct PriorWeights([f32; sanko_search::PRIOR_FEATURES]);

fn parse_scale(s: &str) -> Result<f32, String> {
    match s.parse::<f32>() {
        Ok(v) if v.is_finite() && v >= 0.0 => Ok(v),
        _ => Err(format!("not a finite number >= 0: {s}")),
    }
}

fn parse_l2(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(format!("not a number > 0: {s}")),
    }
}

fn parse_gib(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(v) if v.is_finite() && v > 0.0 => Ok(v),
        _ => Err(format!("not a size > 0: {s}")),
    }
}

/// Proof-table buckets for `--proof-gib` on a board of `npoints` points (0 without it).
fn proof_buckets(t: &TableArgs, npoints: usize) -> Result<u64, String> {
    let Some(gib) = t.proof_gib else { return Ok(0) };
    let bucket = if npoints > 32 { 256.0 } else { 192.0 };
    let n = (gib * f64::from(1u32 << 30) / bucket).floor();
    if (1.0..=(1u64 << 34) as f64).contains(&n) {
        Ok(n as u64)
    } else {
        Err(format!("--proof-gib {gib}: {n} buckets, outside 1 to 2^34"))
    }
}

fn parse_prior_weights(s: &str) -> Result<PriorWeights, String> {
    let v: Vec<f32> = s
        .split(',')
        .map(|x| x.trim().parse::<f32>().ok().filter(|w| w.is_finite()).ok_or_else(|| format!("not a finite number: {x}")))
        .collect::<Result<_, _>>()?;
    // Fewer weights than features: the rest are 0 (models trained on fewer features).
    if v.len() > sanko_search::PRIOR_FEATURES {
        return Err(format!("at most {} weights, got {}", sanko_search::PRIOR_FEATURES, v.len()));
    }
    let mut w = [0.0; sanko_search::PRIOR_FEATURES];
    w[..v.len()].copy_from_slice(&v);
    Ok(PriorWeights(w))
}

#[derive(Args, Debug, Clone)]
struct SelftestArgs {
    #[command(flatten)]
    board: BoardArgs,
    /// Number of random positions.
    #[arg(long, value_name = "N", default_value_t = 200, help_heading = "Test")]
    positions: u64,
    /// Seed of the random positions.
    #[arg(long, value_name = "SEED", default_value_t = 0x00C0_FFEE, help_heading = "Test")]
    rng_seed: u64,
    /// Minimum length of the random move sequences (long ones keep big boards in reach).
    #[arg(long, value_name = "N", default_value_t = 0, help_heading = "Test")]
    min_plies: u32,
    /// The oracle also uses no static knowledge (no pure-eye leaves, no region solver).
    #[arg(long, help_heading = "Test")]
    oracle_no_static: bool,
    #[command(flatten)]
    solver: SolverArgs,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
struct StaticcheckArgs {
    #[command(flatten)]
    board: BoardArgs,
    /// Number of random positions.
    #[arg(long, value_name = "N", default_value_t = 200, help_heading = "Test")]
    positions: u64,
    /// Seed of the random positions.
    #[arg(long, value_name = "SEED", default_value_t = 0x00C0_FFEE, help_heading = "Test")]
    rng_seed: u64,
    #[command(flatten)]
    solver: SolverArgs,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
struct PskcheckArgs {
    #[command(flatten)]
    board: BoardArgs,
    /// The moves to replay, comma-separated (Go coordinates, row-column digits or pass).
    #[arg(long, value_name = "MOVES", help_heading = "Position")]
    moves: String,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Args, Debug, Clone)]
struct BenchArgs {
    #[command(flatten)]
    board: BoardArgs,
    #[command(flatten)]
    position: PositionArgs,
    #[command(flatten)]
    solver: SolverArgs,
    #[command(flatten)]
    tables: TableArgs,
    /// Number of benchmark positions.
    #[arg(long, value_name = "N", default_value_t = 20, help_heading = "Benchmark")]
    bench_n: u64,
    /// Random legal stone moves played after --moves for each position.
    #[arg(long, value_name = "N", default_value_t = 6, help_heading = "Benchmark")]
    bench_plies: u32,
    /// Seed of the random moves.
    #[arg(long, value_name = "SEED", default_value_t = 0x00C0_FFEE, help_heading = "Benchmark")]
    rng_seed: u64,
    /// Node limit per position (0 = none).
    #[arg(long, value_name = "N", default_value_t = 20_000_000, help_heading = "Benchmark")]
    node_limit: u64,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    #[value(alias = "b")]
    Black,
    #[value(alias = "w")]
    White,
    None,
}

impl Side {
    const fn color(self) -> Option<Color> {
        match self {
            Self::Black => Some(Color::Black),
            Self::White => Some(Color::White),
            Self::None => None,
        }
    }
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum ColorWhen {
    Auto,
    Always,
    Never,
}

impl OutputArgs {
    /// Apply the colour choice and return whether to format for reading.
    fn pretty(self) -> bool {
        let choice = match self.color {
            ColorWhen::Auto => anstream::ColorChoice::Auto,
            ColorWhen::Always => anstream::ColorChoice::Always,
            ColorWhen::Never => anstream::ColorChoice::Never,
        };
        choice.write_global();
        !self.plain && std::io::stdout().is_terminal()
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        None => match (cli.board.rows, cli.board.cols) {
            (Some(rows), Some(cols)) => solve_cmd(BoardArgs { rows, cols }, &cli.solve),
            _ => Cli::command()
                .error(ErrorKind::MissingRequiredArgument, "--rows and --cols are required to solve a position")
                .exit(),
        },
        Some(Command::Solve(a)) => solve_cmd(a.board, &a.options),
        Some(Command::Selftest(a)) => selftest_cmd(&a),
        Some(Command::Staticcheck(a)) => staticcheck_cmd(&a),
        Some(Command::Pskcheck(a)) => pskcheck_cmd(&a),
        Some(Command::Bench(a)) => bench_cmd(&a),
        Some(Command::Train(a)) => train_cmd(&a),
    }
}

/// Fit the learned prior to proof dumps (see `TrainArgs`).
fn train_cmd(a: &TrainArgs) -> ExitCode {
    use sanko_search::prior::{RECORD_FLOATS, evaluate, fit};
    let label = RECORD_FLOATS - 2;
    let black_share = |d: &[f32]| {
        let n = d.len() / RECORD_FLOATS;
        d.chunks_exact(RECORD_FLOATS).filter(|r| r[label] > 0.5).count() as f64 / n.max(1) as f64
    };
    let mut sets = Vec::new();
    for path in &a.dumps {
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => return fail(&format!("cannot read {}: {e}", path.display())),
        };
        // A run killed while writing leaves a partial last record: drop it.
        let whole = bytes.len() - bytes.len() % (4 * RECORD_FLOATS);
        if whole < bytes.len() {
            eprintln!("note: {}: ignoring a partial last record ({} bytes)", path.display(), bytes.len() - whole);
        }
        let data: Vec<f32> =
            bytes[..whole].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        println!(
            "{}: {} records, Black wins {:.3}",
            path.display(),
            data.len() / RECORD_FLOATS,
            black_share(&data)
        );
        sets.push(data);
    }
    let mean_quality = |w: &[f32; sanko_search::PRIOR_FEATURES], parts: &[&[f32]]| {
        let n = parts.len().max(1) as f64;
        parts.iter().map(|d| evaluate(w, d)).fold((0.0, 0.0), |(l, c), (x, y)| (l + x / n, c + y / n))
    };
    if a.validate && sets.len() > 1 {
        println!("leave one dump out (log loss and accuracy, then the same for the training dumps' base rate):");
        let mut total = (0.0, 0.0);
        for (i, held) in sets.iter().enumerate() {
            let rest: Vec<&[f32]> = sets.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, d)| d.as_slice()).collect();
            let w = fit(&rest, a.l2);
            let (loss, acc) = evaluate(&w, held);
            let base = (rest.iter().map(|d| black_share(d)).sum::<f64>() / rest.len() as f64).clamp(1e-6, 1.0 - 1e-6);
            let mut constant = [0.0f32; sanko_search::PRIOR_FEATURES];
            constant[0] = (base / (1.0 - base)).ln() as f32;
            let (closs, cacc) = evaluate(&constant, held);
            println!("  {}: {loss:.3}, {acc:.3}   (constant {closs:.3}, {cacc:.3})", a.dumps[i].display());
            total = (total.0 + loss, total.1 + closs);
        }
        let n = sets.len() as f64;
        println!("  mean log loss {:.3} (constant {:.3})", total.0 / n, total.1 / n);
    }
    let all: Vec<&[f32]> = sets.iter().map(Vec::as_slice).collect();
    let w = fit(&all, a.l2);
    let (loss, acc) = mean_quality(&w, &all);
    println!("fitted to all dumps: log loss {loss:.3}, accuracy {acc:.3}");
    println!("--prior-weights {}", w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(","));
    ExitCode::SUCCESS
}

/// Run `f::<R, C, 1, R·C>(args…)` for a compiled board size, `None` otherwise.
macro_rules! with_board {
    ($rows:expr, $cols:expr, $f:ident $args:tt) => {
        with_board!(@sizes ($rows, $cols) $f $args;
            (1,4) (1,5) (1,6) (1,7) (1,8) (1,9) (1,10) (1,11) (1,12) (1,13) (1,14) (1,15) (1,16)
            (1,17) (1,18) (1,19) (1,20) (1,21) (1,22) (1,23) (1,24)
            (2,2) (2,3) (2,4) (2,5) (2,6) (2,7) (2,8) (2,9) (2,10) (2,11) (2,12) (2,13) (2,14) (2,15)
            (3,3) (3,4) (3,5) (3,6) (3,7) (3,8) (3,9) (3,10)
            (4,4) (4,5) (4,6) (4,7) (5,5) (5,6))
    };
    (@sizes ($rows:expr, $cols:expr) $f:ident $args:tt; $(($r:literal, $c:literal))*) => {
        match ($rows, $cols) {
            $( ($r, $c) => Some($f::<$r, $c, 1, { $r * $c }> $args), )*
            _ => None,
        }
    };
}

const SIZES: &str = "1x4 to 1x24, 2x2 to 2x15, 3x3 to 3x10, 4x4 to 4x7, 5x5 and 5x6";

fn fail(msg: &str) -> ExitCode {
    ui::error(msg);
    ExitCode::from(2)
}

fn unsupported(b: BoardArgs) -> ExitCode {
    fail(&format!("board size {}x{} is not compiled in (supported: {SIZES})", b.rows, b.cols))
}

/// First line of every plain log: the exact build and command line.
fn banner() {
    println!("sanko {VERSION} (source {SOURCE}): {}", std::env::args().skip(1).collect::<Vec<_>>().join(" "));
}

fn komi_times_two(komi: f64) -> i32 {
    (komi * 2.0).round() as i32
}

/// Komi is a multiple of 0.5 (it is solved as twice its value, an integer).
fn parse_komi(s: &str) -> Result<f64, String> {
    let k: f64 = s.parse().map_err(|_| format!("not a number: {s}"))?;
    if !k.is_finite() || k.abs() > 1000.0 || (k * 2.0).fract() != 0.0 {
        return Err(format!("komi must be a multiple of 0.5, got {s}"));
    }
    Ok(k)
}

fn solver_config(s: &SolverArgs) -> DfpnConfig {
    if s.prior == PriorKind::Learned && (s.eye_init > 0 || s.eye_area || s.eye_weight > 0 || s.eye_komi) {
        eprintln!("note: --eye-init/--eye-area/--eye-weight/--eye-komi apply only with --prior eye (ignored)");
    }
    DfpnConfig {
        eps: s.eps,
        prune_for: s.prune_for.and_then(Side::color),
        threads: s.threads,
        exact_only: s.exact_only,
        eye_init: s.eye_init,
        eye_area: s.eye_area,
        region_solver: s.regions && !s.no_regions,
        prior_weights: match s.prior {
            PriorKind::Learned => Some(s.prior_weights.map_or(sanko_search::prior::DEFAULT_PRIOR_WEIGHTS, |w| w.0)),
            PriorKind::Eye => None,
        },
        prior_scale: s.prior_scale,
        eye_weight: s.eye_weight,
        eye_komi: s.eye_komi,
        virtual_loss: !s.no_virtual_loss,
        prefetch: !s.no_prefetch,
        keep_tempo: s.keep_tempo,
        region_interior: s.region_interior,
        ..DfpnConfig::default()
    }
}

fn with_tables(cfg: DfpnConfig, t: &TableArgs) -> DfpnConfig {
    DfpnConfig {
        tt_log_buckets: t.pn_tt_log,
        proof_log: t.proof_log.unwrap_or_else(|| t.pn_tt_log.saturating_sub(1)),
        static_cache_log_slots: t.sc_log,
        ..cfg
    }
}

/// Parse a comma-separated move list (Go coordinates, row-column digits, pass).
fn parse_moves<const R: usize, const C: usize>(s: &str) -> Result<Vec<Move<R, C>>, String> {
    s.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| {
            if t.eq_ignore_ascii_case("pass") || t == "p" {
                return Ok(Move::Pass);
            }
            let b = t.as_bytes();
            if b.len() == 2 && b[0].is_ascii_digit() && b[1].is_ascii_digit() {
                return Point::new((b[0] - b'0') as usize, (b[1] - b'0') as usize)
                    .map(Move::Stone)
                    .ok_or_else(|| format!("move '{t}' is off the {R}x{C} board"));
            }
            ui::parse_coord::<R, C>(t)
                .map(Move::Stone)
                .ok_or_else(|| format!("cannot read move '{t}' on a {R}x{C} board (use C3, 22 or pass)"))
        })
        .collect()
}

/// Play `moves` from the empty board, checking every move.
fn replay<const R: usize, const C: usize, const L: usize, const N: usize>(
    moves: &[Move<R, C>],
) -> Result<(Position<R, C, L, N>, HashHistory), String> {
    let mut pos: Position<R, C, L, N> = Position::new();
    let mut hist = HashHistory::new();
    hist.push(pos.board().hash());
    for (i, &m) in moves.iter().enumerate() {
        if pos.is_terminal() {
            return Err(format!("move {} ({}) comes after the game has ended", i + 1, ui::move_name(m)));
        }
        pos.play(m, &mut hist).map_err(|e| {
            let why = match e {
                PlayError::Occupied => "the point is occupied",
                PlayError::Suicide => "it is suicide",
                PlayError::PositionalSuperko => "it repeats an earlier board (positional superko)",
            };
            format!("move {} ({}) is illegal: {why}", i + 1, ui::move_name(m))
        })?;
    }
    Ok((pos, hist))
}

/// Row-column digits of a move (the plain format).
fn rc<const R: usize, const C: usize>(m: Move<R, C>) -> String {
    match m {
        Move::Pass => "pass".to_string(),
        Move::Stone(p) => format!("{}{}", p.row(), p.col()),
    }
}

const fn color_name(c: Color) -> &'static str {
    match c {
        Color::Black => "Black",
        Color::White => "White",
    }
}

// ---------------------------------------------------------------------------
// solve

fn solve_cmd(board: BoardArgs, a: &SolveOptions) -> ExitCode {
    let pretty = a.output.pretty();
    sanko_core::set_atari_dead(a.solver.atari_dead);
    with_board!(board.rows, board.cols, solve_on(a, pretty)).unwrap_or_else(|| unsupported(board))
}

fn solve_on<const R: usize, const C: usize, const L: usize, const N: usize>(a: &SolveOptions, pretty: bool) -> ExitCode {
    let moves = match parse_moves::<R, C>(&a.position.moves) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    let (root, hist) = match replay::<R, C, L, N>(&moves) {
        Ok(x) => x,
        Err(e) => return fail(&e),
    };
    let mut cfg = with_tables(solver_config(&a.solver), &a.tables);
    cfg.komi_times_two = komi_times_two(a.position.komi);
    cfg.node_limit = a.node_limit;
    cfg.progress = a.progress;
    let live = pretty && std::io::stderr().is_terminal();
    cfg.proof_buckets = match proof_buckets(&a.tables, N) {
        Ok(n) => n,
        Err(e) => return fail(&e),
    };
    let entry = if N > 32 { 256u64 } else { 192 };
    let proof_bytes = entry * if cfg.proof_buckets > 0 { cfg.proof_buckets } else { 1 << cfg.proof_log };
    if live {
        cfg.progress = true;
        cfg.on_progress = Some(panel::progress);
        panel::start(panel::Context {
            rows: R,
            cols: C,
            cells: (0..N).map(|i| root.board().at(Point::<R, C>::from_index_unchecked(i))).collect(),
            bytes: [64u64 << cfg.tt_log_buckets, proof_bytes, 8u64 << cfg.static_cache_log_slots],
        });
    }
    let table_bytes = (64u64 << cfg.tt_log_buckets) + proof_bytes;
    if pretty {
        solve_header::<R, C, L, N>(a, &cfg, &root, &moves, table_bytes + (8u64 << cfg.static_cache_log_slots));
    } else {
        banner();
        println!(
            "solving {R}x{C} komi={} (tt 2^{} buckets = {} MB, eps={}, prune_for={:?}, threads={}, prior={}) ...",
            a.position.komi,
            cfg.tt_log_buckets,
            table_bytes >> 20,
            cfg.eps,
            cfg.prune_for,
            cfg.threads,
            prior_label(&cfg),
        );
    }
    if let Some(path) = &a.dump_proofs {
        if let Err(e) = sanko_search::enable_proof_dump(path, a.dump_rate) {
            return fail(&format!("cannot write {path}: {e}"));
        }
    }
    let started = Instant::now();
    let r = solve_dfpn::<R, C, L, N>(&moves, cfg);
    sanko_search::flush_proof_dump();
    let el = started.elapsed().as_secs_f64();
    if live {
        panel::clear();
    }
    if pretty {
        solve_report::<R, C, L, N>(a, &cfg, &root, &hist, &r, el);
    } else {
        solve_plain::<R, C, L, N>(&root, &hist, &r, el);
    }
    ExitCode::SUCCESS
}

/// Replay move codes from `root`: the final position, the last stone move, the
/// moves played, and whether every code was legal.
fn follow<const R: usize, const C: usize, const L: usize, const N: usize>(
    root: &Position<R, C, L, N>,
    hist: &HashHistory,
    codes: &[usize],
) -> (Position<R, C, L, N>, Option<Point<R, C>>, Vec<Move<R, C>>, bool) {
    let mut pos = root.clone();
    let mut hist = hist.clone();
    let mut last = None;
    let mut moves = Vec::new();
    for &code in codes {
        let Some(m) = Move::<R, C>::decode(code) else { break };
        if pos.play(m, &mut hist).is_err() {
            return (pos, last, moves, false);
        }
        if let Move::Stone(p) = m {
            last = Some(p);
        }
        moves.push(m);
    }
    (pos, last, moves, true)
}

fn solve_plain<const R: usize, const C: usize, const L: usize, const N: usize>(
    root: &Position<R, C, L, N>,
    hist: &HashHistory,
    r: &DfpnResult,
    el: f64,
) {
    if !r.pv.is_empty() {
        let (end, _, moves, ok) = follow(root, hist, &r.pv);
        let mut text: Vec<String> = moves.iter().map(|&m| rc(m)).collect();
        if !ok {
            text.push("(illegal!)".to_string());
        }
        println!("pv ({} plies): {}", text.len(), text.join(" "));
        println!("pv end: {end:?}");
    }
    if let Some(m) = r.root_move.and_then(Move::<R, C>::decode) {
        println!("root winning move: {}", rc(m));
    }
    let s = &r.stats;
    println!(
        "winner: {:?}   nodes: {}   children: {}   elapsed: {:.3}s   nodes/s: {:.0}",
        r.winner,
        s.nodes,
        s.children,
        el,
        s.nodes as f64 / el.max(1e-9)
    );
    println!(
        "leaves: terminal={} static={}   tt: probes={} hits={}   max_ply={}   sym_merged={}",
        s.terminal_leaves, s.static_leaves, s.tt_probes, s.tt_hits, s.max_ply, s.sym_merged
    );
    println!(
        "ghi: proofs_stored={} reuse same-path={} (rejected {}) checked={}  reject target={} bloom={}  twin hits={} stores={} frame-rejects={}  psk_exclusions={}",
        s.proofs_stored,
        s.reuse_exact,
        s.reject_same_path,
        s.reuse_checked,
        s.reject_target,
        s.reject_bloom,
        s.twin_hits,
        s.twin_stores,
        s.twin_frame_rejects,
        s.psk_exclusions
    );
    println!(
        "static: rejected (superko-unsafe on path)={}  transfers rejected by territory={}  territory-pruned moves={}",
        s.static_rejects, s.reject_terr, s.terr_pruned
    );
    println!("128-bit proof check: mismatches={}  proofs with one target: {}", s.check_mismatch, s.proofs_with_target);
    let p = s.path_specific;
    println!(
        "path-specific proofs: two-targets={}  symmetry-merge={}  from-table={}  alternates: twin-child={}  base-taken={}  stored={} reused={}",
        p[0], p[1], p[2], p[3], p[4], s.alt_stores, s.alt_hits
    );
    println!(
        "closed-region solver leaves: {}  failures [unused, too big, <2 pure eyes, local loss]: {:?}",
        s.region_leaves, s.reg_fail
    );
    println!(
        "first-visit expansions: {} of {} ({:.2}x revisit factor)",
        s.fresh,
        s.nodes,
        s.nodes as f64 / s.fresh.max(1) as f64
    );
}

// One-line summaries of the solver options, for the pretty headers.

/// The static and local knowledge in use, one entry per component.
fn knowledge_parts(cfg: &DfpnConfig) -> Vec<String> {
    let mut parts = vec!["pure-eye life and territory".to_string()];
    if cfg.region_solver {
        parts.push(if cfg.region_interior > 0 {
            format!("closed regions, interior moves up to {} points", cfg.region_interior)
        } else {
            "closed regions".to_string()
        });
    }
    if cfg.prior_weights.is_some() {
        let which = if cfg.prior_weights == Some(sanko_search::prior::DEFAULT_PRIOR_WEIGHTS) { "built-in" } else { "custom" };
        parts.push(format!("learned prior ({which} weights, scale {})", cfg.prior_scale));
    } else if cfg.eye_init > 0 {
        let mut prior = format!("eye prior T={}", cfg.eye_init);
        if cfg.eye_area {
            prior.push_str(", area");
        }
        if cfg.eye_weight > 0 {
            let _ = write!(prior, ", weight {}", cfg.eye_weight);
        }
        if cfg.eye_komi {
            prior.push_str(", komi-aware");
        }
        parts.push(prior);
    }
    parts
}

/// The source of the initial numbers, for the plain banner.
fn prior_label(cfg: &DfpnConfig) -> String {
    match cfg.prior_weights {
        Some(w) if w == sanko_search::prior::DEFAULT_PRIOR_WEIGHTS => format!("learned(scale {})", cfg.prior_scale),
        Some(_) => format!("learned(custom weights, scale {})", cfg.prior_scale),
        None => "eye".to_string(),
    }
}

/// The knowledge in use on one line, for the solver summaries.
fn knowledge_text(cfg: &DfpnConfig) -> String {
    knowledge_parts(cfg).join(" · ")
}

fn pruning_text(cfg: &DfpnConfig) -> String {
    match cfg.prune_for {
        None => "none (both sides searched in full)".to_string(),
        Some(c) => {
            let mut t = format!("{}: no own eye fills or moves in own territory", color_name(c));
            if cfg.keep_tempo {
                t.push_str(", one tempo move kept");
            }
            t
        }
    }
}

/// Pruning in a few words, for the one-line solver summaries.
fn pruning_short(cfg: &DfpnConfig) -> String {
    match cfg.prune_for {
        None => "no pruning".to_string(),
        Some(c) => format!("pruned for {}{}", color_name(c), if cfg.keep_tempo { " (tempo kept)" } else { "" }),
    }
}

fn search_text(cfg: &DfpnConfig, bytes: u64) -> String {
    let mut t = format!(
        "df-pn · ε {} · {} thread{} · tables {}",
        cfg.eps,
        cfg.threads,
        if cfg.threads == 1 { "" } else { "s" },
        ui::bytes(bytes)
    );
    if cfg.exact_only {
        t.push_str(" · no proof transfer (exact only)");
    }
    if !cfg.virtual_loss && cfg.threads > 1 {
        t.push_str(" · no virtual loss");
    }
    t
}

fn solve_header<const R: usize, const C: usize, const L: usize, const N: usize>(
    a: &SolveOptions,
    cfg: &DfpnConfig,
    root: &Position<R, C, L, N>,
    moves: &[Move<R, C>],
    bytes: u64,
) {
    ui::header("solve", VERSION, SOURCE);
    ui::row("Command", ui::paint(ui::DIM, std::env::args().skip(1).collect::<Vec<_>>().join(" ")));
    ui::row(
        "Position",
        format!("{R}×{C} · {} to move · komi {}", color_name(root.to_move()), a.position.komi),
    );
    ui::row(
        "Moves",
        if moves.is_empty() {
            ui::paint(ui::DIM, "none (empty board)")
        } else {
            format!("{}  {}", ui::move_list(moves, 18), ui::paint(ui::DIM, format!("({} moves)", moves.len())))
        },
    );
    ui::row("Search", search_text(cfg, bytes));
    ui::row_lines("Knowledge", &knowledge_parts(cfg));
    ui::row("Pruning", pruning_text(cfg));
    if a.solver.atari_dead {
        ui::row("Rules", ui::paint(ui::WARN, "MIGOS one-liberty convention (not positional-superko Go)"));
    }
    anstream::println!();
}

fn solve_report<const R: usize, const C: usize, const L: usize, const N: usize>(
    a: &SolveOptions,
    cfg: &DfpnConfig,
    root: &Position<R, C, L, N>,
    hist: &HashHistory,
    r: &DfpnResult,
    el: f64,
) {
    let s = &r.stats;
    if let Some(w) = r.winner {
        anstream::println!(
            "  {}  {}   {}",
            ui::paint(ui::GOOD, ui::stone(w)),
            ui::paint(ui::GOOD, format!("{} wins", color_name(w))),
            ui::paint(ui::DIM, format!("at komi {} · proved in {}", a.position.komi, ui::duration(el)))
        );
        if cfg.prune_for.is_some() {
            anstream::println!(
                "     {}",
                ui::paint(ui::DIM, "valid under pruning: only the winner's own moves were skipped")
            );
        }
    } else {
        let why = if cfg.node_limit > 0 && s.nodes >= cfg.node_limit {
            format!("stopped at the node limit of {} expansions", ui::grouped(cfg.node_limit))
        } else if let Some(c) = cfg.prune_for {
            format!(
                "no proof that {} wins; with --prune-for {} only a win for {} is reported",
                color_name(c),
                color_name(c).to_lowercase(),
                color_name(c)
            )
        } else {
            "the search gave up without a result".to_string()
        };
        anstream::println!("  {}  {}   {}", ui::paint(ui::WARN, "?"), ui::paint(ui::WARN, "Unknown"), ui::paint(ui::DIM, why));
    }
    anstream::println!();
    if s.nodes == 0 {
        anstream::println!(
            "  {}",
            ui::paint(ui::DIM, "decided at the root by the rules or static knowledge, without search")
        );
        anstream::println!();
        ui::board(root, None, &[]);
        anstream::println!();
        return;
    }
    if let Some(m) = r.root_move.and_then(Move::<R, C>::decode) {
        ui::row("First move", ui::paint(ui::LABEL, ui::move_name(m)));
    }
    let (end, last, line, legal) = follow(root, hist, &r.pv);
    if !legal {
        ui::row("Proof line", ui::paint(ui::WARN, "illegal move in the proof line"));
    }
    if line.is_empty() {
        ui::board(root, None, &[]);
    } else {
        ui::row(
            "Proof line",
            format!("{}  {}", ui::move_list(&line, 24), ui::paint(ui::DIM, format!("({} plies)", line.len()))),
        );
        anstream::println!();
        anstream::println!("  {}", ui::paint(ui::DIM, "after the proof line:"));
        ui::board(&end, last, &[]);
    }
    anstream::println!();
    ui::row(
        "Effort",
        format!(
            "{} expansions · {}/s · revisit ×{:.2} · depth {}",
            ui::grouped(s.nodes),
            ui::compact(s.nodes as f64 / el.max(1e-9)),
            s.nodes as f64 / s.fresh.max(1) as f64,
            s.max_ply
        ),
    );
    ui::row(
        "Proofs",
        format!(
            "{} stored · reused {} on the same path, {} transferred, {} path-specific",
            ui::grouped(s.proofs_stored),
            ui::grouped(s.reuse_exact),
            ui::grouped(s.reuse_checked),
            ui::grouped(s.twin_hits)
        ),
    );
    ui::row(
        "Rejected",
        format!(
            "{} by target · {} by Bloom · {} by territory · {} mirror frame",
            ui::grouped(s.reject_target),
            ui::grouped(s.reject_bloom),
            ui::grouped(s.reject_terr),
            ui::grouped(s.twin_frame_rejects)
        ),
    );
    ui::row(
        "Leaves",
        format!(
            "{} terminal · {} static · {} closed-region · {} superko exclusions",
            ui::grouped(s.terminal_leaves),
            ui::grouped(s.static_leaves),
            ui::grouped(s.region_leaves),
            ui::grouped(s.psk_exclusions)
        ),
    );
    ui::row(
        "Integrity",
        format!("{} proof entries rejected by the second hash", ui::grouped(s.check_mismatch)),
    );
    anstream::println!();
}

/// Print a pass/fail summary line.
fn verdict(ok: bool, summary: &str) {
    if ok {
        anstream::println!("  {}  {summary}", ui::paint(ui::GOOD, "✔ passed"));
    } else {
        anstream::println!("  {}  {summary}", ui::paint(ui::BAD, "✘ failed"));
    }
    anstream::println!();
}

// ---------------------------------------------------------------------------
// selftest

fn selftest_cmd(a: &SelftestArgs) -> ExitCode {
    let pretty = a.output.pretty();
    sanko_core::set_atari_dead(a.solver.atari_dead);
    if !pretty {
        banner();
    }
    match with_board!(a.board.rows, a.board.cols, selftest_on(a, pretty)) {
        Some(true) => ExitCode::SUCCESS,
        Some(false) => ExitCode::from(1),
        None => unsupported(a.board),
    }
}

fn selftest_on<const R: usize, const C: usize, const L: usize, const N: usize>(a: &SelftestArgs, pretty: bool) -> bool {
    let n = a.positions;
    let base0 = solver_config(&a.solver);
    if pretty {
        ui::header("selftest", VERSION, SOURCE);
        ui::row(
            "Positions",
            format!("{n} random positions on {R}×{C} · seed {} · at least {} plies", a.rng_seed, a.min_plies),
        );
        ui::row("Solver", format!("{} · {}", knowledge_text(&base0), pruning_short(&base0)));
        ui::row(
            "Oracle",
            if a.oracle_no_static { "exact-path reuse only, no static knowledge" } else { "exact-path reuse only" },
        );
        anstream::println!();
    }
    let live = pretty && std::io::stderr().is_terminal();
    let mut rng = Rng::new(a.rng_seed);
    let mut mismatches = 0u64;
    let mut solved = 0u64;
    let started = Instant::now();
    for i in 0..n {
        // Random legal prefix (stones mostly, an occasional pass). Long
        // prefixes keep big-board positions within the oracle's reach.
        let mut pos: Position<R, C, L, N> = Position::new();
        let mut hist = HashHistory::new();
        hist.push(pos.board().hash());
        let min = a.min_plies.min(N as u32);
        let len = min + rng.next_below((N as u32) + 4 - min);
        let mut moves: Vec<Move<R, C>> = Vec::new();
        for _ in 0..len {
            let mut legal: Vec<Move<R, C>> = (0..N)
                .map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k)))
                .filter(|&m| pos.legal(m, &hist))
                .collect();
            if legal.is_empty() || (rng.next_below(10) == 0 && moves.len() as u32 >= min) {
                legal.push(Move::Pass);
            }
            let mv = legal[rng.next_below(legal.len() as u32) as usize];
            if pos.play(mv, &mut hist).is_err() {
                break;
            }
            moves.push(mv);
            if pos.is_terminal() || pos.consecutive_passes() > 0 {
                break;
            }
        }
        if pos.is_terminal() {
            continue;
        }
        let k2 = 2 * (rng.next_below(N as u32 + 1) as i32 - (N as i32) / 2) + 1;
        let base = DfpnConfig {
            komi_times_two: k2,
            tt_log_buckets: 14,
            proof_log: 13,
            static_cache_log_slots: 16,
            node_limit: 20_000_000,
            progress: false,
            ..base0
        };
        let full = solve_dfpn::<R, C, L, N>(&moves, base);
        let oracle = solve_dfpn::<R, C, L, N>(
            &moves,
            DfpnConfig {
                exact_only: true,
                threads: 1,
                region_solver: false,
                prune_for: None,
                no_static: a.oracle_no_static,
                ..base
            },
        );
        if let (Some(x), Some(y)) = (full.winner, oracle.winner) {
            solved += 1;
            if x != y {
                mismatches += 1;
                if pretty {
                    ui::clear_status();
                    anstream::println!(
                        "  {}  position #{i}: komi {}, moves {}: solver says {}, oracle says {}",
                        ui::paint(ui::BAD, "✘ mismatch"),
                        f64::from(k2) / 2.0,
                        ui::move_list(&moves, 60),
                        color_name(x),
                        color_name(y)
                    );
                } else {
                    println!("MISMATCH #{i}: komi2={k2} moves={moves:?} full={:?} exact={:?}", full.winner, oracle.winner);
                }
            }
        }
        if live {
            ui::status(&format!(
                "{}/{n} positions · {solved} compared · {mismatches} mismatches · {}",
                i + 1,
                ui::duration(started.elapsed().as_secs_f64())
            ));
        } else if !pretty && (i + 1) % 25 == 0 {
            println!(
                "[selftest {R}x{C}] {}/{n} positions, {solved} compared, {mismatches} mismatches, {:.1}s",
                i + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    if pretty {
        ui::clear_status();
        verdict(
            mismatches == 0,
            &format!(
                "{n} positions · {solved} compared · {mismatches} mismatches · {}",
                ui::duration(started.elapsed().as_secs_f64())
            ),
        );
    } else {
        println!("[selftest {R}x{C}] done: {solved} compared, {mismatches} mismatches");
    }
    mismatches == 0
}

// ---------------------------------------------------------------------------
// staticcheck

fn staticcheck_cmd(a: &StaticcheckArgs) -> ExitCode {
    let pretty = a.output.pretty();
    sanko_core::set_atari_dead(a.solver.atari_dead);
    if !pretty {
        banner();
    }
    match with_board!(a.board.rows, a.board.cols, staticcheck_on(a, pretty)) {
        Some(true) => ExitCode::SUCCESS,
        Some(false) => ExitCode::from(1),
        None => unsupported(a.board),
    }
}

fn staticcheck_on<const R: usize, const C: usize, const L: usize, const N: usize>(
    a: &StaticcheckArgs,
    pretty: bool,
) -> bool {
    let n = a.positions;
    let base0 = DfpnConfig {
        tt_log_buckets: 16,
        proof_log: 15,
        static_cache_log_slots: 16,
        node_limit: 30_000_000,
        ..solver_config(&a.solver)
    };
    if pretty {
        ui::header("staticcheck", VERSION, SOURCE);
        ui::row("Positions", format!("{n} random positions on {R}×{C} · seed {}", a.rng_seed));
        ui::row(
            "Solver",
            format!("{} · {}", knowledge_text(&DfpnConfig { region_solver: true, ..base0 }), pruning_short(&base0)),
        );
        ui::row("Oracle", "exact-path reuse only, no static knowledge");
        anstream::println!();
    }
    let live = pretty && std::io::stderr().is_terminal();
    let mut rng = Rng::new(a.rng_seed);
    let (mut probes, mut agreed, mut unknown, mut bad) = (0u64, 0u64, 0u64, 0u64);
    let started = Instant::now();
    for i in 0..n {
        let mut pos: Position<R, C, L, N> = Position::new();
        let mut hist = HashHistory::new();
        hist.push(pos.board().hash());
        let len = rng.next_below(2 * N as u32 + 2);
        let mut moves: Vec<Move<R, C>> = Vec::new();
        let mut cap = [0u64; 2];
        for _ in 0..len {
            let mut legal: Vec<Move<R, C>> = (0..N)
                .map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k)))
                .filter(|&m| pos.legal(m, &hist))
                .collect();
            if legal.is_empty() || rng.next_below(12) == 0 {
                legal.push(Move::Pass);
            }
            let mv = legal[rng.next_below(legal.len() as u32) as usize];
            let player = pos.to_move();
            let Ok(undo) = pos.play(mv, &mut hist) else { break };
            cap[player.opponent() as usize] |= undo.captures.bits();
            moves.push(mv);
            if pos.consecutive_passes() > 0 {
                break;
            }
        }
        if pos.is_terminal() {
            continue;
        }
        let analysis = sanko_core::pure_eye_analysis(pos.board());
        let points = N as i32;
        let mut cases: Vec<(Color, i32, &str)> = Vec::new();
        for color in [Color::Black, Color::White] {
            let (alive, terr) = match color {
                Color::Black => (analysis.black_alive, analysis.black_territory),
                Color::White => (analysis.white_alive, analysis.white_territory),
            };
            let s = (alive.count() + terr.count()) as i32;
            if s == 0 || cap[color as usize] & terr.bits() != 0 {
                continue;
            }
            let k2 = match color {
                Color::Black => 4 * s - 2 * points - 1,
                Color::White => 2 * points - 4 * s + 1,
            };
            cases.push((color, k2, "static"));
        }
        if !analysis.black_alive.is_empty() {
            cases.push((Color::Black, 2 * points - 1, "kill-all"));
        }
        for (color, k2, kind) in cases {
            let cfg = DfpnConfig { komi_times_two: k2, ..base0 };
            let full = solve_dfpn::<R, C, L, N>(&moves, DfpnConfig { region_solver: true, ..cfg });
            let oracle = solve_dfpn::<R, C, L, N>(
                &moves,
                DfpnConfig { exact_only: true, no_static: true, region_solver: false, prune_for: None, threads: 1, ..cfg },
            );
            probes += 1;
            match (full.winner, oracle.winner) {
                (_, None) | (None, _) => unknown += 1,
                (Some(x), Some(y)) if x == y => agreed += 1,
                (Some(x), Some(y)) => {
                    bad += 1;
                    if pretty {
                        ui::clear_status();
                        anstream::println!(
                            "  {}  position #{i} ({kind} probe for {}): komi {}, moves {}: solver says {}, oracle says {}",
                            ui::paint(ui::BAD, "✘ counterexample"),
                            color_name(color),
                            f64::from(k2) / 2.0,
                            ui::move_list(&moves, 60),
                            color_name(x),
                            color_name(y)
                        );
                        ui::board(&pos, None, &[]);
                    } else {
                        println!(
                            "COUNTEREXAMPLE #{i} ({kind}, claimed {color:?}): komi2={k2} full={x:?} oracle={y:?} moves={moves:?}\n{pos:?}"
                        );
                    }
                }
            }
            if kind == "static" && full.winner.is_some() && full.winner != Some(color) {
                if pretty {
                    ui::clear_status();
                    anstream::println!(
                        "  {}  position #{i}: the static rule said {} but the solver says {}",
                        ui::paint(ui::WARN, "note"),
                        color_name(color),
                        color_name(full.winner.unwrap_or(Color::White))
                    );
                } else {
                    println!("NOTE #{i}: static said {color:?} but full solver says {:?}", full.winner);
                }
            }
        }
        if live {
            ui::status(&format!(
                "{}/{n} positions · {probes} probes · {agreed} agreed · {unknown} unknown · {bad} counterexamples · {}",
                i + 1,
                ui::duration(started.elapsed().as_secs_f64())
            ));
        } else if !pretty && (i + 1) % 50 == 0 {
            println!(
                "[staticcheck {R}x{C}] {}/{n} positions, {probes} probes, {agreed} agreed, {unknown} unknown, {bad} counterexamples, {:.1}s",
                i + 1,
                started.elapsed().as_secs_f64()
            );
        }
    }
    if pretty {
        ui::clear_status();
        verdict(
            bad == 0,
            &format!(
                "{probes} probes · {agreed} agreed · {unknown} unknown · {bad} counterexamples · {}",
                ui::duration(started.elapsed().as_secs_f64())
            ),
        );
    } else {
        println!("[staticcheck {R}x{C}] done: {probes} probes, {agreed} agreed, {unknown} unknown, {bad} counterexamples");
    }
    bad == 0
}

// ---------------------------------------------------------------------------
// pskcheck

fn pskcheck_cmd(a: &PskcheckArgs) -> ExitCode {
    let pretty = a.output.pretty();
    if !pretty {
        banner();
    }
    with_board!(a.board.rows, a.board.cols, pskcheck_on(a, pretty)).unwrap_or_else(|| unsupported(a.board))
}

/// Replay the moves. At every position, list the stone moves that positional
/// superko forbids but basic ko would allow (they do not recreate the board
/// from just before the opponent's last move).
fn pskcheck_on<const R: usize, const C: usize, const L: usize, const N: usize>(a: &PskcheckArgs, pretty: bool) -> ExitCode {
    let moves = match parse_moves::<R, C>(&a.moves) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    if let Err(e) = replay::<R, C, L, N>(&moves) {
        return fail(&e);
    }
    if pretty {
        ui::header("pskcheck", VERSION, SOURCE);
        ui::row(
            "Moves",
            format!("{}  {}", ui::move_list(&moves, 40), ui::paint(ui::DIM, format!("({} moves)", moves.len()))),
        );
        anstream::println!();
    }
    let mut pos: Position<R, C, L, N> = Position::new();
    let mut hist = HashHistory::new();
    hist.push(pos.board().hash());
    let mut boards: Vec<u64> = vec![pos.board().hash()];
    let mut last = None;
    let mut found = 0;
    for (ply, &m) in moves.iter().enumerate() {
        let before_opp = if boards.len() >= 2 { Some(boards[boards.len() - 2]) } else { None };
        let mut only_psk = Vec::new();
        for k in 0..N {
            let p = Point::<R, C>::from_index_unchecked(k);
            if let Some(h) = pos.board_hash_after(p) {
                if hist.contains(h) && Some(h) != before_opp {
                    only_psk.push(p);
                }
            }
        }
        if !only_psk.is_empty() {
            found += 1;
            if pretty {
                let names: Vec<String> = only_psk.iter().map(|&p| ui::coord(p)).collect();
                anstream::println!(
                    "  {}  {} to move · superko forbids {} {}",
                    ui::paint(ui::LABEL, format!("Move {:>3}", ply + 1)),
                    color_name(pos.to_move()),
                    ui::paint(ui::BAD, names.join(" ")),
                    ui::paint(ui::DIM, "(basic ko would allow it)")
                );
                ui::board(&pos, last, &only_psk);
                anstream::println!();
            } else {
                let names: Vec<String> = only_psk.iter().map(|p| format!("{}{}", p.row(), p.col())).collect();
                println!("ply {ply} ({:?} to move): superko-only exclusions: {}", pos.to_move(), names.join(" "));
                println!("{pos:?}");
            }
        }
        pos.play(m, &mut hist).expect("checked by replay");
        if let Move::Stone(p) = m {
            boards.push(pos.board().hash());
            last = Some(p);
        }
    }
    if pretty {
        if found == 0 {
            anstream::println!("  {}", ui::paint(ui::GOOD, "No move along the line is forbidden by superko alone."));
            anstream::println!();
        }
        anstream::println!("  {}", ui::paint(ui::DIM, "final position:"));
        ui::board(&pos, last, &[]);
        anstream::println!();
    } else {
        println!("final: {pos:?}");
    }
    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// bench

fn bench_cmd(a: &BenchArgs) -> ExitCode {
    let pretty = a.output.pretty();
    sanko_core::set_atari_dead(a.solver.atari_dead);
    if !pretty {
        banner();
    }
    with_board!(a.board.rows, a.board.cols, bench_on(a, pretty)).unwrap_or_else(|| unsupported(a.board))
}

fn bench_on<const R: usize, const C: usize, const L: usize, const N: usize>(a: &BenchArgs, pretty: bool) -> ExitCode {
    let prefix = match parse_moves::<R, C>(&a.position.moves) {
        Ok(m) => m,
        Err(e) => return fail(&e),
    };
    let (root, root_hist) = match replay::<R, C, L, N>(&prefix) {
        Ok(x) => x,
        Err(e) => return fail(&e),
    };
    let buckets = match proof_buckets(&a.tables, N) {
        Ok(n) => n,
        Err(e) => return fail(&e),
    };
    let cfg = DfpnConfig {
        komi_times_two: komi_times_two(a.position.komi),
        node_limit: a.node_limit,
        progress: false,
        proof_buckets: buckets,
        ..with_tables(solver_config(&a.solver), &a.tables)
    };
    if pretty {
        ui::header("bench", VERSION, SOURCE);
        ui::row(
            "Suite",
            format!(
                "{} positions on {R}×{C} · {} random plies after {} · seed {} · komi {}",
                a.bench_n,
                a.bench_plies,
                if prefix.is_empty() { "the empty board".to_string() } else { ui::move_list(&prefix, 12) },
                a.rng_seed,
                a.position.komi
            ),
        );
        ui::row("Solver", format!("{} · {}", knowledge_text(&cfg), pruning_short(&cfg)));
        anstream::println!();
        anstream::println!(
            "  {}",
            ui::paint(ui::LABEL, format!("{:>4}  {:<8} {:>14} {:>12}   moves", "#", "winner", "expansions", "time"))
        );
    }
    let live = pretty && std::io::stderr().is_terminal();
    let mut rng = Rng::new(a.rng_seed);
    let (mut solved, mut total_nodes, mut total_secs) = (0u64, 0u64, 0.0f64);
    for i in 0..a.bench_n {
        let mut pos = root.clone();
        let mut hist = root_hist.clone();
        let mut moves = prefix.clone();
        for _ in 0..a.bench_plies {
            let legal: Vec<Move<R, C>> = (0..N)
                .map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k)))
                .filter(|&m| pos.legal(m, &hist))
                .collect();
            if legal.is_empty() {
                break;
            }
            let mv = legal[rng.next_below(legal.len() as u32) as usize];
            pos.play(mv, &mut hist).expect("legal");
            moves.push(mv);
        }
        if live {
            ui::status(&format!("solving position {} of {} …", i + 1, a.bench_n));
        }
        let started = Instant::now();
        let r = solve_dfpn::<R, C, L, N>(&moves, cfg);
        let secs = started.elapsed().as_secs_f64();
        // Finished below the cap (a win against the pruned side is reported
        // as `None` but is still a finished search).
        let ok = r.winner.is_some() || cfg.node_limit == 0 || r.stats.nodes < cfg.node_limit;
        if ok {
            solved += 1;
            total_nodes += r.stats.nodes;
            total_secs += secs;
        }
        if pretty {
            ui::clear_status();
            let winner = match r.winner {
                Some(w) => ui::paint(ui::GOOD, format!("{:<8}", color_name(w))),
                None if ok => ui::paint(ui::DIM, format!("{:<8}", "unknown")),
                None => ui::paint(ui::WARN, format!("{:<8}", "limit")),
            };
            anstream::println!(
                "  {i:>4}  {winner} {:>14} {:>12}   {}",
                ui::grouped(r.stats.nodes),
                ui::duration(secs),
                ui::paint(ui::DIM, ui::move_list(&moves[prefix.len()..], 12))
            );
        } else {
            println!(
                "bench #{i}: winner={:?} nodes={} secs={secs:.2} moves={}",
                r.winner,
                r.stats.nodes,
                moves.iter().map(|&m| rc(m)).collect::<Vec<_>>().join(",")
            );
        }
    }
    if pretty {
        anstream::println!();
        ui::row(
            "Total",
            format!("{solved}/{} finished · {} expansions · {}", a.bench_n, ui::grouped(total_nodes), ui::duration(total_secs)),
        );
        anstream::println!();
    } else {
        println!("bench total: solved={solved}/{} nodes={total_nodes} secs={total_secs:.1}", a.bench_n);
    }
    ExitCode::SUCCESS
}
