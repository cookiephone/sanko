//! Depth-first proof-number search (df-pn) over the transposition DAG, with a
//! sound, cheap treatment of the GHI problem under positional superko.
//!
//! ## Why df-pn
//!
//! A best-first search tree keeps every explored-but-unsolved node live. On
//! 4×4 that is a multi-GB frontier and on 4×5 it exhausts memory. df-pn
//! (Nagai 2002) keeps only the current root→leaf path on the stack and *all*
//! other search state (proof/disproof numbers of unproven nodes and the
//! proofs themselves) in a fixed-size transposition table, so memory is
//! bounded by the table, and transpositions are shared rather than
//! re-expanded.
//!
//! ## Negamax formulation
//!
//! Every node stores `(phi, delta)` from the side to move's view: `phi` is the
//! proof number of "side to move wins", `delta` that of "side to move loses".
//! `phi(n) = min_c delta(c)`, `delta(n) = Σ_c phi(c)`. Solved: win = `(0, ∞)`,
//! loss = `(∞, 0)`. The descent is standard df-pn (`mid`), optionally with the
//! 1+ε trick on the selected child's threshold.
//!
//! ## GHI under positional superko
//!
//! A position's TT key ignores history, but superko legality does not. A proof
//! `T` of node `N` computed with path history `H` transfers to another path `H'`
//! iff
//!
//! 1. no *winner* move of `T` becomes illegal under `H'` (i.e. no board of `T`
//!    is an ancestor on `H'`), and
//! 2. no *loser* move that was excluded under `H` (because it repeated an
//!    ancestor of `N`) becomes legal under `H'` (i.e. every such *exclusion
//!    target* board is still on `H'`).
//!
//! (Within `T` the internal path is identical on both histories, and terminal
//! and pure-eye static leaves are history-independent.) Each proof therefore carries
//! attributes (see [`crate::pntt`]): a Bloom filter over the canonical hashes of
//! its boards (checks 1 conservatively, since a Bloom miss is a proof of absence),
//! whether it has an exclusion target (any surviving target makes the proof
//! path-specific: it is stored under the twin key only), the fewest stones on any proof board (an
//! ancestor with fewer stones cannot be a proof board), and, for a path-specific
//! proof, the set-signature of the path it was proved on. Attributes compose bottom-up exactly:
//! a won node's proof is its winning child's plus itself, and a lost node's is the
//! union of all children's plus itself, plus its own superko exclusions.
//!
//! A target-free proof (base or alternate slot) is reused only through the
//! transfer test, which uses exact data only (exact board hashes in the Bloom
//! filter, exact stone sets and captures), on its own path as on any other: no
//! step relies on a hash being collision-free.
//!
//! Proofs whose exclusions are path-specific are stored under a **twin** key
//! (`position key ^ path signature`). This is Kishimoto & Müller's base/twin split,
//! with two independent non-linear set hashes of the path's boards (`mix` and
//! `mix2` of each board, XOR'd over the path). The twin key carries the first,
//! and the entry's second hash is sealed with the second, so a twin proof is
//! reused only on a path whose board set agrees with its own in both 64-bit
//! signatures (the identical superko game up to a 2^-128 accident). When a base
//! proof fails to transfer, the node is searched under its twin key on that path.
//!
//! Base slots are write-once. A target-free proof that finds its base slot
//! taken by another path's proof, or that was proved for a child searched under
//! its twin key, is stored under one of two **alternate** base keys
//! (`key ^ ALT_KEYS[i]`, chosen by the path signature) rather than the twin key.
//! Any path can reuse it there through the same 128-bit identity and transfer
//! test as a base proof, while a twin-key proof serves only the identical game.
//!
//! Symmetry: sibling moves are merged only under symmetries that fix *every*
//! board on the path (then the superko games are isomorphic too). A lost node
//! whose proof relied on such a merge is marked path-specific.
//!
//! ## Initial numbers
//!
//! A new child's proof and disproof numbers come from the learned prior (a
//! logistic model of the probability p that the side to move wins, over cheap
//! board features, see [`crate::prior`]: `1 + S·(−ln p)` and `1 + S·(−ln(1−p))`)
//! or from the eye-space prior (White's black-free area). Either only orders the
//! search: an unsolved child's numbers are always in `[1, 2^40]`, and a child is
//! solved only by a terminal, static, table or searched proof.
//!
//! ## Pruning the conjectured winner's moves
//!
//! With `prune_for = Some(c)`, side `c` never fills its own single-point eyes
//! and never plays inside its own certified territory (unless a `c` stone was
//! captured there on the path). `keep_tempo` keeps one such move per node.
//! Dropping a *winner's* option can never make a proof of `c` winning invalid,
//! but it makes disproofs of `c` unsound, so a root result against `c` is then
//! reported as unknown.

use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sanko_core::{
    Board, Color, HashHistory, Move, PASS_KEY, PlayError, Point, Position, STM_BLACK, STM_WHITE,
    Symmetry, terminal_winner,
};

use crate::checkpoint::{self, CheckpointError, CheckpointHeader, CheckpointOptions};
use crate::pntt::{INF as TT_INF, PnEntry, PnStatus, PnTable, TableFill};

/// Solved proof/disproof number. Proof numbers are 64-bit inside the search:
/// transposition double counting compounds through the DAG and overflows 32
/// bits on 3×5-sized problems, and a *saturated* sum can then never exceed the
/// root threshold (a livelock). The TT keeps 15-bit floating-point copies
/// (rounded down, never 0 or INF for an unsolved node).
const INF: u64 = u64::MAX;
/// Largest unsolved proof number.
const MAX_UNSOLVED: u64 = 1 << 62;

#[inline]
const fn from_tt(v: u64) -> u64 {
    v
}

#[inline]
const fn to_tt(v: u64) -> u64 {
    if v == INF { TT_INF } else if v < MAX_UNSOLVED { v } else { MAX_UNSOLVED }
}
use crate::static_cache::StaticCache;

/// Number of evaluation features (see `Searcher::prior_features`).
pub const PRIOR_FEATURES: usize = 14;

/// Sampled proofs for training the evaluation: one record per sampled proof,
/// `PRIOR_FEATURES` little-endian `f32` features, then `1.0` if Black wins
/// (else `0.0`), then log2 of the proof's work.
static PROOF_DUMP: std::sync::OnceLock<(u64, std::sync::Mutex<std::io::BufWriter<std::fs::File>>)> =
    std::sync::OnceLock::new();

/// Write about one proof in `rate` (chosen by its key, every proof for 0 or
/// 1) to `path` (see `PROOF_DUMP`). Once per process.
///
/// # Errors
/// If a dump is already enabled (the file is then left alone), or the file
/// cannot be created.
pub fn enable_proof_dump(path: &str, rate: u64) -> std::io::Result<()> {
    if PROOF_DUMP.get().is_some() {
        return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "a proof dump is already enabled"));
    }
    let f = std::io::BufWriter::new(std::fs::File::create(path)?);
    PROOF_DUMP
        .set((rate.max(1), std::sync::Mutex::new(f)))
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::AlreadyExists, "a proof dump is already enabled"))
}

/// Flush the proof dump, if any.
pub fn flush_proof_dump() {
    if let Some((_, w)) = PROOF_DUMP.get()
        && let Ok(mut w) = w.lock()
    {
        let _ = std::io::Write::flush(&mut *w);
    }
}

/// Atari count, two-liberty count and total liberties of the chains of `stones`.
fn chain_stats<const COLS: usize, const NPOINTS: usize>(stones: u64, empties: u64) -> (u32, u32, u32) {
    let (mut atari, mut two, mut libs) = (0, 0, 0);
    let mut rest = stones;
    while rest != 0 {
        let mut comp = rest & rest.wrapping_neg();
        loop {
            let g = sanko_core::dilate_bits::<COLS, NPOINTS>(comp) & stones;
            if g == comp {
                break;
            }
            comp = g;
        }
        rest &= !comp;
        let l = (sanko_core::neighbors_bits::<COLS, NPOINTS>(comp) & empties).count_ones();
        atari += u32::from(l == 1);
        two += u32::from(l == 2);
        libs += l;
    }
    (atari, two, libs)
}

/// Configuration for [`solve_dfpn`].
#[derive(Copy, Clone, Debug)]
pub struct DfpnConfig {
    /// Learned prior: weights of a logistic model over `Searcher::prior_features`
    /// giving the probability that Black wins (see [`crate::prior`]). When set, it
    /// replaces the eye-space prior for the initial numbers (it never decides a node).
    pub prior_weights: Option<[f32; PRIOR_FEATURES]>,
    /// Scale of the learned prior: initial numbers are `1 + scale · (−ln p)`.
    pub prior_scale: f32,
    /// Doubled komi (e.g. 3 for 1.5). A draw (integer komi) counts as a White win.
    pub komi_times_two: i32,
    /// log₂ of the number of unproven-entry TT buckets (8 × 8 B each).
    pub tt_log_buckets: u32,
    /// log₂ of the number of proof-table buckets (4 entries of 48 B, or 64 B on
    /// boards above 32 points).
    pub proof_log: u32,
    /// Number of proof-table buckets, any value from 1 to 2^34, in place of
    /// `2^proof_log` (0 = use `proof_log`).
    pub proof_buckets: u64,
    /// log₂ of the static-safety cache slots (0 = off).
    pub static_cache_log_slots: u32,
    /// 1+ε trick: the selected child's delta threshold is
    /// `max(δ₂+1, ⌈δ₂·(1+ε)⌉)`. `0.0` = plain df-pn.
    pub eps: f64,
    /// Stop after this many node expansions (0 = unlimited) → unknown result.
    pub node_limit: u64,
    /// Conjectured winner whose own eye-filling moves, and moves inside its own
    /// certified territory, are pruned (sound only for proofs that this side wins).
    pub prune_for: Option<Color>,
    /// Print periodic progress lines to stderr.
    pub progress: bool,
    /// Worker threads sharing the transposition table (1 = serial).
    pub threads: usize,
    /// Reuse a stored proof only on the identical superko game (same position,
    /// same set of path boards), which is GHI-safe by construction. A differential
    /// oracle for the transfer filters (much slower).
    pub exact_only: bool,
    /// Kill-everything leaf prior from White's eye potential (threshold `T`,
    /// 0 = off). Intended for maximal-komi Black-kills-all problems.
    pub eye_init: u32,
    /// `eye_init` counts White stones with no Black neighbour as eye space too.
    pub eye_area: bool,
    /// Closed-region local solver for Black (see `Searcher::regions_won`).
    pub region_solver: bool,
    /// Closed-region solver: largest region (points) it analyses (cost only:
    /// any limit is sound).
    pub region_max: u32,
    /// Closed-region solver: in regions of at most this many points, Black may
    /// also play interior (non-anchored) points in the local game. Any White
    /// capture of a Black stone counts as a local loss, and a region is won
    /// only once all its Black stones are connected to the walls (see
    /// `local_black_wins_interior`). 0 = off.
    pub region_interior: u32,
    /// Disable all static knowledge (pure-eye leaves, territory pruning, region
    /// solver): leaves are game ends, plus the pass-back shortcut (the side to
    /// move after an opponent's pass ends the game by passing). An oracle for
    /// the static rules (for testing, very slow).
    pub no_static: bool,
    /// With `prune_for`: keep the first pruned own-eye / own-territory move of
    /// each node (a board-changing tempo move, which under PSK is not
    /// equivalent to a pass). Still sound for the pruned side's proofs.
    pub keep_tempo: bool,
    /// Kawano's simulation, with this node budget per attempt (0 = off). Once a
    /// child is proven a win for the opponent, its siblings are tried the same
    /// way: the opponent's stored winning moves of that proof are replayed
    /// against every reply (see `simulate`). Each sibling is tried at most once
    /// per visit of the node: when a sibling is proven, or when it is selected
    /// for search while untried, from the latest proven sibling.
    pub simulate: u64,
    /// Prefetch the TT and static-cache lines of all stone children before
    /// classifying them, so their cache misses overlap (speed only).
    pub prefetch: bool,
    /// Virtual loss: a child with `b` other workers inside is selected by
    /// `delta · (1 + b)`. Off only for the ablation (selection by `delta`).
    pub virtual_loss: bool,
    /// `eye_init`: extra area credited per existing White eye point.
    pub eye_weight: u32,
    /// `eye_init`: measure White's area above the most it may own while Black
    /// still wins at this komi (0 for kill-all problems), instead of above 0.
    pub eye_komi: bool,
    /// With `progress`: called with a snapshot instead of printing progress
    /// lines (more often: every 2^18 expansions instead of 2^23, at most 4
    /// times a second). Reporting only, the search is unaffected.
    pub on_progress: Option<fn(&Progress)>,
}

/// A progress snapshot of a running solve (see [`DfpnConfig::on_progress`]).
#[derive(Clone, Debug)]
pub struct Progress {
    /// Seconds since the search started.
    pub elapsed: f64,
    /// Node expansions so far, all threads.
    pub nodes: u64,
    /// The root's proof and disproof numbers (side to move's view).
    pub root_phi: u64,
    pub root_delta: u64,
    /// Deepest ply reached by the reporting thread.
    pub max_ply: u64,
    /// The root's moves as `(move code, phi, delta)`, each seen from the
    /// position after the move: delta 0 marks a winning move, phi 0 a refuted
    /// one. A move code is a point index (row-major, row 0 at the top) or the
    /// number of points for a pass. Empty until the root was expanded once.
    pub root_moves: Vec<(usize, u64, u64)>,
    /// Point maps of the symmetries that fix every board of the game so far.
    /// A root move that a listed map sends onto a listed move was merged with
    /// it and has the same value.
    pub root_symmetries: Vec<Vec<usize>>,
    /// Sampled occupancy of the transposition table.
    pub table_fill: TableFill,
    /// Sampled fraction of the static-cache slots in use.
    pub static_fill: f64,
}

impl Default for DfpnConfig {
    fn default() -> Self {
        Self {
            komi_times_two: 3,
            tt_log_buckets: 20,
            proof_log: 18,
            static_cache_log_slots: 22,
            eps: 0.15,
            node_limit: 0,
            prune_for: None,
            progress: false,
            threads: 1,
            exact_only: false,
            eye_init: 0,
            eye_area: false,
            region_solver: false,
            eye_weight: 0,
            eye_komi: false,
            virtual_loss: true,
            prefetch: true,
            keep_tempo: false,
            simulate: 100,
            no_static: false,
            region_max: 14,
            region_interior: 0,
            on_progress: None,
            prior_weights: None,
            prior_scale: crate::prior::DEFAULT_PRIOR_SCALE,
            proof_buckets: 0,
        }
    }
}

/// Search statistics.
#[derive(Copy, Clone, Debug, Default)]
pub struct DfpnStats {
    /// `mid` invocations (node expansions, counting revisits).
    pub nodes: u64,
    /// Children generated and classified.
    pub children: u64,
    pub terminal_leaves: u64,
    pub static_leaves: u64,
    pub tt_probes: u64,
    pub tt_hits: u64,
    /// Proven base or alternate entries that passed the transfer test.
    pub reuse_checked: u64,
    /// Proven base entries rejected as path-specific (`many`) or by `--exact-only`.
    pub reject_target: u64,
    /// Proven base entries rejected by the Bloom ancestor check.
    pub reject_bloom: u64,
    /// Proven twin entries reused.
    pub twin_hits: u64,
    /// Proofs stored under a twin key.
    pub twin_stores: u64,
    /// Twin entries rejected because they belong to a mirror image of the
    /// current board with the same set of earlier boards (another game).
    pub twin_frame_rejects: u64,
    /// Superko-illegal moves encountered during generation.
    pub psk_exclusions: u64,
    /// Sibling moves merged by path symmetry.
    pub sym_merged: u64,
    /// Deepest ply reached.
    pub max_ply: u64,
    pub proofs_stored: u64,
    /// Proofs stored with exactly one superko exclusion target.
    pub proofs_with_target: u64,
    /// Static verdicts not used as leaves (superko-unsafe on this path).
    pub static_rejects: u64,
    /// Proof transfers rejected by the static-territory condition.
    pub reject_terr: u64,
    /// Pruned-side moves skipped (own eye fills and moves inside its own territory).
    pub terr_pruned: u64,
    /// Proof entries whose 64-bit key matched but second hash did not: twin-key
    /// collisions or slots mixed by concurrent writes, which this count cannot
    /// tell apart (base and alternate keys cannot collide, see `PnEntry::check`).
    /// In a parallel run it is an upper bound on true collisions.
    pub check_mismatch: u64,
    /// Expansions of nodes that had no TT entry (first visits, modulo eviction).
    pub fresh: u64,
    /// Static Black wins certified by the closed-region solver.
    pub region_leaves: u64,
    /// Closed-region failures: [unused, region too big, a chain without two
    /// pure eyes, local game lost].
    pub reg_fail: [u64; 4],
    /// Proofs not stored in their base slot, by cause. Path-specific (twin key):
    /// [two different targets, symmetry merge, inherited from a path-specific
    /// entry] (these can overlap). Alternates (reusable by any path): [proof of
    /// a child searched under its twin key, base slot taken].
    pub path_specific: [u64; 5],
    /// Proofs stored under an alternate base key, and alternates reused.
    pub alt_stores: u64,
    pub alt_hits: u64,
    /// Simulation: attempts, successes, nodes spent.
    pub sim_attempts: u64,
    pub sim_successes: u64,
    pub sim_nodes: u64,
}

impl DfpnStats {
    fn merge(&mut self, o: &Self) {
        self.nodes += o.nodes;
        self.children += o.children;
        self.terminal_leaves += o.terminal_leaves;
        self.static_leaves += o.static_leaves;
        self.tt_probes += o.tt_probes;
        self.tt_hits += o.tt_hits;
        self.reuse_checked += o.reuse_checked;
        self.reject_target += o.reject_target;
        self.reject_bloom += o.reject_bloom;
        self.twin_hits += o.twin_hits;
        self.twin_stores += o.twin_stores;
        self.twin_frame_rejects += o.twin_frame_rejects;
        self.psk_exclusions += o.psk_exclusions;
        self.sym_merged += o.sym_merged;
        self.max_ply = self.max_ply.max(o.max_ply);
        self.proofs_stored += o.proofs_stored;
        self.proofs_with_target += o.proofs_with_target;
        self.static_rejects += o.static_rejects;
        self.reject_terr += o.reject_terr;
        self.terr_pruned += o.terr_pruned;
        self.check_mismatch += o.check_mismatch;
        self.fresh += o.fresh;
        self.region_leaves += o.region_leaves;
        for i in 0..4 {
            self.reg_fail[i] += o.reg_fail[i];
        }
        for i in 0..5 {
            self.path_specific[i] += o.path_specific[i];
        }
        self.alt_stores += o.alt_stores;
        self.alt_hits += o.alt_hits;
        self.sim_attempts += o.sim_attempts;
        self.sim_successes += o.sim_successes;
        self.sim_nodes += o.sim_nodes;
    }
}

/// Checkpoint gate state (see `Shared::park`).
#[derive(Debug, Default)]
struct Gate {
    /// A snapshot is being taken.
    paused: bool,
    /// Workers waiting in `Shared::park`.
    parked: usize,
    /// Workers that have not returned yet.
    active: usize,
}

/// State shared by all worker threads besides the TT and static cache.
#[derive(Debug)]
struct Shared {
    /// Per-key count of workers currently inside a node (virtual-loss table).
    busy: Vec<AtomicU16>,
    busy_mask: u64,
    /// Set once the root is solved (or the node limit is hit): all unwind.
    stop: AtomicBool,
    /// Node expansions across all workers (updated in batches).
    nodes: AtomicU64,
    /// The first root verdict found: `true` = side to move at the root wins.
    result: Mutex<Option<bool>>,
    /// Set while a checkpoint is taken: workers park at their next batch check
    /// (a fast copy of `gate.paused`, see `Shared::park`).
    pause: AtomicBool,
    gate: Mutex<Gate>,
    gate_cv: Condvar,
}

impl Shared {
    fn new(threads: usize) -> Self {
        let log = if threads > 1 { 20 } else { 0 };
        let n = 1usize << log;
        Self {
            busy: (0..n).map(|_| AtomicU16::new(0)).collect(),
            busy_mask: (n as u64) - 1,
            stop: AtomicBool::new(false),
            nodes: AtomicU64::new(0),
            result: Mutex::new(None),
            pause: AtomicBool::new(false),
            gate: Mutex::new(Gate::default()),
            gate_cv: Condvar::new(),
        }
    }

    /// Wait while a checkpoint is taken (workers call this at their batch check
    /// once `pause` is set). The worker keeps its stack and continues after.
    fn park(&self) {
        let mut g = self.gate.lock().expect("gate lock");
        g.parked += 1;
        self.gate_cv.notify_all();
        while g.paused {
            g = self.gate_cv.wait(g).expect("gate lock");
        }
        g.parked -= 1;
    }

    /// A worker has returned (see `Leaving`).
    fn leave(&self) {
        self.gate.lock().expect("gate lock").active -= 1;
        self.gate_cv.notify_all();
    }

    /// Ask every worker to park and wait until all running workers have, so
    /// that the table stays unchanged. Returns whether any worker still runs.
    fn pause_all(&self) -> bool {
        let mut g = self.gate.lock().expect("gate lock");
        g.paused = true;
        self.pause.store(true, Ordering::Relaxed);
        while g.parked < g.active {
            g = self.gate_cv.wait(g).expect("gate lock");
        }
        g.active > 0
    }

    /// Let the parked workers continue.
    fn resume_all(&self) {
        let mut g = self.gate.lock().expect("gate lock");
        g.paused = false;
        self.pause.store(false, Ordering::Relaxed);
        drop(g);
        self.gate_cv.notify_all();
    }

    /// Wait until every worker has returned, at most `timeout`. Returns
    /// whether all have.
    fn wait_done(&self, timeout: Duration) -> bool {
        let g = self.gate.lock().expect("gate lock");
        self.gate_cv.wait_timeout_while(g, timeout, |g| g.active > 0).expect("gate lock").0.active == 0
    }

    #[inline]
    fn slot(&self, key: u64) -> &AtomicU16 {
        &self.busy[((key >> 17) & self.busy_mask) as usize]
    }
}

/// Outcome of [`solve_dfpn`].
#[derive(Clone, Debug)]
pub struct DfpnResult {
    /// Proven winner from the root, or `None` (node limit hit, or a disproof of
    /// the pruned side that cannot be trusted).
    pub winner: Option<Color>,
    pub stats: DfpnStats,
    /// When the side to move at the root wins: the stored winning move's code
    /// (point index, or `NPOINTS` for pass), if its proof is still in the table.
    pub root_move: Option<usize>,
    /// A principal variation of the proof (move codes from the root): the
    /// winner's stored moves and the loser's most resistant replies.
    pub pv: Vec<usize>,
}

/// Closed-region local game state (see `Searcher::local_black_wins`): full
/// bitboards, so the key is exact on any board size.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct RegionKey {
    region: u64,
    a0: u64,
    white: u64,
    fills: u64,
    white_to_move: bool,
}

/// Multiply-rotate hasher for the region cache (std's `SipHash` showed up as
/// ~12% of a 5×5 profile). Not DoS-resistant: the keys are our own bitboards.
#[derive(Default, Debug)]
struct RegionHash(u64);

impl std::hash::Hasher for RegionHash {
    #[inline]
    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(23) ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }
}

type RegionHasher = std::hash::BuildHasherDefault<RegionHash>;

const MAX_PLY: usize = 1000;

#[inline]
const fn mix(x: u64) -> u64 {
    // splitmix64 finalizer: a bijective non-linear mix, so XOR-ing mixed board
    // hashes is a set hash with no exploitable linear structure.
    let mut z = x ^ 0x5851_F42D_4C95_7F2D;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A second bijective non-linear mix, independent of `mix` (murmur3 finalizer
/// constants): the XOR of `mix2` over the path's boards is a second set hash.
#[inline]
const fn mix2(x: u64) -> u64 {
    let mut z = x ^ 0xD6E8_FEB8_6659_FD93;
    z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    z ^ (z >> 33)
}

/// The two words of the blocked 128-bit Bloom filter of a board with canonical
/// hash `canon`: two bits in the first or the second word. They are taken from
/// the mixed hash, since the canonical hash is the least of the symmetric
/// images' hashes and its top bits lean towards zero.
#[inline]
const fn bloom_pair(canon: u64) -> (u64, u64) {
    let m = mix(canon);
    let b = (1u64 << (m >> 58)) | (1u64 << ((m >> 52) & 63));
    if (m >> 40) & 1 == 0 { (b, 0) } else { (0, b) }
}

/// Superko exclusion targets of a proof (raw-frame board hashes of ancestors
/// whose presence removed a loser move).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
enum Target {
    #[default]
    None,
    One(u64),
    /// Two or more targets, or a symmetry-merge dependency: path-specific. The
    /// bits record why (statistics only): `MANY_OVERFLOW`, `MANY_MERGE`, `MANY_TABLE`.
    Many(u8),
}

/// Two different exclusion targets met.
const MANY_OVERFLOW: u8 = 1;
/// A loss whose proof relied on a symmetry merge of sibling moves.
const MANY_MERGE: u8 = 2;
/// Inherited from a path-specific table entry (the table keeps no targets).
const MANY_TABLE: u8 = 4;

/// Alternate base keys (`key ^ ALT_KEYS[i]`): a target-free proof whose base slot
/// already holds another path's proof is stored under one of them, so that other
/// paths can still find it. An alternate is a full proof of the same position
/// with its own attributes, accepted only through the 128-bit identity and the
/// transfer test, exactly like a base proof.
/// Its key can collide with another position's key, as a twin key can, and the
/// second hash rejects such a hit.
const ALT_KEYS: [u64; 2] = [0x8F3C_6A1E_5B27_D94B, 0x3B9D_1F05_E6A4_C877];

/// Second-hash keys (`Searcher::key2`): side to move and the pass state.
const K2_STM_BLACK: u64 = 0x6A09_E667_F3BC_C908;
const K2_STM_WHITE: u64 = 0xBB67_AE85_84CA_A73B;
const K2_PASS: u64 = 0x3C6E_F372_FE94_F82B;

/// Second-hash key of a stone of colour index `c` (0 Black, 1 White) on point `i`.
const fn k2_point(i: usize, c: usize) -> u64 {
    mix(0xA076_1D64_78BD_642F ^ ((i * 2 + c) as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB))
}

impl Target {
    const fn union(self, o: Self) -> Self {
        match (self, o) {
            (Self::None, x) | (x, Self::None) => x,
            (Self::One(a), Self::One(b)) if a == b => Self::One(a),
            (Self::One(_), Self::One(_)) => Self::Many(MANY_OVERFLOW),
            (Self::Many(a), Self::Many(b)) => Self::Many(a | b),
            (Self::Many(a), Self::One(_)) | (Self::One(_), Self::Many(a)) => Self::Many(a),
        }
    }

    /// Cause bits of a `Many` (0 otherwise).
    const fn many_causes(self) -> u8 {
        match self {
            Self::Many(c) => c,
            _ => 0,
        }
    }

    const fn without(self, h: u64) -> Self {
        match self {
            Self::One(a) if a == h => Self::None,
            x => x,
        }
    }
}

/// GHI attributes of a proof (see module docs), over the boards produced by
/// the proof's *winner*, since only those can make a transferred proof invalid (a
/// repeated loser-produced board merely removes a loser option). `ab`/`aw` are
/// the black and white stones present on every such board, in the current frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Attrs {
    bloom: u64,
    /// Second word of the blocked 128-bit Bloom filter (see `bloom_pair`).
    bloom2: u64,
    target: Target,
    min_stones: u8,
    ab: u64,
    aw: u64,
    /// Union of the winner's territory over the proof's static leaves.
    terr: u64,
}

impl Attrs {
    /// The attributes of a node that is not proven, paired only with nonzero
    /// numbers (an abort, a threshold, a child's initial value). On a proof they
    /// would be unsound: the all-zero Bloom filter claims "no winner boards" and
    /// never rejects a transfer. No proof carries them: with no winner boards
    /// the common-stone sets are full, and every winner board sets Bloom bits.
    const UNPROVEN: Self = Self { bloom: 0, bloom2: 0, target: Target::None, min_stones: 0, ab: 0, aw: 0, terr: 0 };

    /// No boards (identity of `join`).
    const fn empty<const NPOINTS: usize>() -> Self {
        let full = if NPOINTS >= 64 { u64::MAX } else { (1u64 << NPOINTS) - 1 };
        Self {
            bloom: 0,
            bloom2: 0,
            target: Target::None,
            min_stones: u8::MAX,
            ab: full,
            aw: full,
            terr: 0,
        }
    }

    const fn join(self, o: Self) -> Self {
        Self {
            bloom: self.bloom | o.bloom,
            bloom2: self.bloom2 | o.bloom2,
            target: self.target.union(o.target),
            min_stones: if self.min_stones < o.min_stones { self.min_stones } else { o.min_stones },
            ab: self.ab & o.ab,
            aw: self.aw & o.aw,
            terr: self.terr | o.terr,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Child {
    /// `Move::encode` of the move.
    mv: u8,
    phi: u64,
    delta: u64,
    /// Effective TT key (base or twin).
    key: u64,
    /// Canonical position key.
    base_key: u64,
    twin: bool,
    attrs: Attrs,
    /// No TT entry existed when the child was generated.
    fresh: bool,
}

#[derive(Copy, Clone, Debug, Default)]
struct PathEntry {
    canon: u64,
    stones: u8,
    black: u64,
    white: u64,
    /// Points where a stone of each color (`Color as usize`) was captured
    /// somewhere on the path up to this board.
    cap: [u64; 2],
}

struct Searcher<'a, const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize> {
    pos: Position<ROWS, COLS, LANES, NPOINTS>,
    hist: HashHistory,
    /// One entry per board on the path (parallel to `hist`'s stack).
    path: Vec<PathEntry>,
    /// Set-signature of the path's boards: XOR of the first term of
    /// [`Self::sig_terms`] (`mix(raw hash)` up to 30 points).
    sig: u64,
    /// Second, independent set-signature: XOR of the second term
    /// (`mix2(raw hash)` up to 30 points).
    sig2: u64,
    /// The current board's hash under each symmetry (index 0 = raw).
    sym_hashes: [u64; 8],
    /// The same under the board's second table (boards above 30 points, zero
    /// below). With `sym_hashes` it forms the exact 128-bit board key of each
    /// symmetric image, which the stabilizer and the path signatures need there.
    sym_hashes2: [u64; 8],
    /// Bit `s` set ⇔ symmetry `s` fixes every board on the path.
    path_sym_mask: u8,
    tt: &'a PnTable,
    sc: &'a StaticCache,
    cfg: DfpnConfig,
    stats: DfpnStats,
    /// Move-generation order (point indices, centre first).
    order: Vec<usize>,
    /// `rank[idx]` = position of `idx` in `order`.
    rank: Vec<usize>,
    /// `sym_map[s][idx]` = index of `idx` under symmetry `s`.
    sym_map: Vec<Vec<usize>>,
    nsym: usize,
    /// Reusable child buffers, one per recursion depth.
    bufs: Vec<Vec<Child>>,
    /// Simulation: the shadow position on the proven sibling's line, its own
    /// history (from the simulation's start only), and child buffers per depth.
    shadow_pos: Position<ROWS, COLS, LANES, NPOINTS>,
    shadow_hist: HashHistory,
    sim_bufs: Vec<Vec<Child>>,
    started: Instant,
    next_report: u64,
    aborted: bool,
    sh: &'a Shared,
    tid: usize,
    par: bool,
    /// Nodes not yet added to `sh.nodes`.
    pending_nodes: u64,
    /// Latest root (phi, delta), for progress lines.
    root_pd: (u64, u64),
    /// Root children `(move code, phi, delta, TT key)` (worker 0, with
    /// progress). Reports refresh the numbers from the shared TT.
    root_kids: Vec<(u8, u64, u64, u64)>,
    /// The symmetry mask of the root (recorded with `root_kids`).
    root_sym_mask: u8,
    /// Seconds into the search before which no progress snapshot is built.
    next_progress: f64,
    /// `(selected move, phi, delta)` of the nodes at depths 0..4 on the current path.
    path_pd: [(u8, u64, u64); 4],
    /// `generate` lists every child, also after a winning one (see
    /// `children_numbers`). Never set during search.
    list_all: bool,
    /// Memo of the closed-region local games (see `regions_won`).
    region_cache: std::collections::HashMap<RegionKey, bool, RegionHasher>,
    /// Second Zobrist table, per symmetry: `zob2[s][i][c]` keys a `c` stone at
    /// `i` as seen under symmetry `s` (see [`Self::key2`]).
    zob2: Vec<[[u64; 2]; 64]>,
}

const SYMS: [Symmetry; 8] = [
    Symmetry::Identity,
    Symmetry::FlipH,
    Symmetry::FlipV,
    Symmetry::Rot180,
    Symmetry::Rot90,
    Symmetry::Rot270,
    Symmetry::DiagMain,
    Symmetry::DiagAnti,
];

/// Index of the inverse of symmetry `s` (in `SYMS` order).
const fn sym_inverse(s: usize) -> usize {
    match s {
        4 => 5,
        5 => 4,
        x => x,
    }
}

impl<'a, const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>
    Searcher<'a, ROWS, COLS, LANES, NPOINTS>
{
    /// A searcher at the position reached by `prefix` from the empty board
    /// (the prefix's boards form the superko history above the root).
    fn new(
        prefix: &[Move<ROWS, COLS>],
        tt: &'a PnTable,
        sc: &'a StaticCache,
        cfg: DfpnConfig,
        sh: &'a Shared,
        tid: usize,
    ) -> Self {
        let mut me = Self::new_at(Position::new(), tt, sc, cfg, sh, tid);
        for &mv in prefix {
            me.do_move(mv).expect("illegal move in prefix");
        }
        me
    }

    fn new_at(
        pos: Position<ROWS, COLS, LANES, NPOINTS>,
        tt: &'a PnTable,
        sc: &'a StaticCache,
        cfg: DfpnConfig,
        sh: &'a Shared,
        tid: usize,
    ) -> Self {
        let nsym = Symmetry::all_for(ROWS, COLS).len();
        let mut order: Vec<usize> = (0..NPOINTS).collect();
        // Centre first: sort by squared distance from the board centre, ties by
        // index (deterministic).
        let dist = |i: usize| {
            let r = (i / COLS) as i64 * 2 - (ROWS as i64 - 1);
            let c = (i % COLS) as i64 * 2 - (COLS as i64 - 1);
            r * r + c * c
        };
        order.sort_by_key(|&i| (dist(i), i));
        let mut rank = vec![0; NPOINTS];
        for (k, &i) in order.iter().enumerate() {
            rank[i] = k;
        }
        let sym_map = SYMS
            .iter()
            .take(nsym)
            .map(|s| (0..NPOINTS).map(|i| s.apply_index(i, ROWS, COLS)).collect())
            .collect();
        let board = *pos.board();
        let mut sym_hashes = [0u64; 8];
        for (s, h) in sym_hashes.iter_mut().enumerate().take(nsym) {
            *h = board.hash_under_symmetry(SYMS[s]);
        }
        let mut sym_hashes2 = [0u64; 8];
        if Board::<ROWS, COLS, LANES, NPOINTS>::WIDE {
            for (s, h) in sym_hashes2.iter_mut().enumerate().take(nsym) {
                *h = board.hash2_under_symmetry(SYMS[s]);
            }
        }
        let mut hist = HashHistory::with_capacity(256);
        hist.push(board.key());
        let (sig, sig2) = Self::sig_terms(board.hash(), board.hash2());
        let shadow_pos = pos.clone();
        let mut me = Self {
            pos,
            shadow_pos,
            shadow_hist: HashHistory::with_capacity(256),
            sim_bufs: Vec::new(),
            hist,
            path: Vec::with_capacity(256),
            sig,
            sig2,
            sym_hashes,
            sym_hashes2,
            path_sym_mask: 0,
            tt,
            sc,
            cfg,
            stats: DfpnStats::default(),
            order,
            rank,
            sym_map,
            nsym,
            bufs: Vec::new(),
            started: Instant::now(),
            next_report: if cfg.on_progress.is_some() { 1 << 18 } else { 1 << 22 },
            aborted: false,
            sh,
            tid,
            par: cfg.threads > 1,
            pending_nodes: 0,
            root_pd: (0, 0),
            root_kids: Vec::new(),
            root_sym_mask: 1,
            next_progress: 0.0,
            path_pd: [(0, 0, 0); 4],
            list_all: false,
            zob2: Vec::new(),
            region_cache: std::collections::HashMap::default(),
        };
        let z2 = k2_point;
        me.zob2 = (0..nsym)
            .map(|s| {
                let mut t = [[0u64; 2]; 64];
                for (i, row) in t.iter_mut().enumerate().take(NPOINTS) {
                    let j = me.sym_map[s][i];
                    *row = [z2(j, 0), z2(j, 1)];
                }
                t
            })
            .collect();
        let (canon, _) = me.canon();
        me.path.push(PathEntry {
            canon,
            stones: board.stone_count() as u8,
            black: board.stones(Color::Black).bits(),
            white: board.stones(Color::White).bits(),
            cap: [0, 0],
        });
        me.path_sym_mask = me.stabilizer();
        me
    }

    /// `(canonical board hash, index of the first symmetry attaining it)`.
    #[inline]
    fn canon(&self) -> (u64, u8) {
        let mut best = self.sym_hashes[0];
        let mut s_best = 0u8;
        for s in 1..self.nsym {
            if self.sym_hashes[s] < best {
                best = self.sym_hashes[s];
                s_best = s as u8;
            }
        }
        (best, s_best)
    }

    /// Bitmask of symmetries fixing the current board.
    #[inline]
    fn stabilizer(&self) -> u8 {
        let mut m = 1u8;
        for s in 1..self.nsym {
            // Exact: the 64-bit hash up to 30 points, the 128-bit key above.
            if self.sym_hashes[s] == self.sym_hashes[0]
                && (!Board::<ROWS, COLS, LANES, NPOINTS>::WIDE || self.sym_hashes2[s] == self.sym_hashes2[0])
            {
                m |= 1 << s;
            }
        }
        m
    }

    /// The two per-board terms of the path signatures. Up to 30 points the
    /// 64-bit board hash is exact and is mixed alone. Above, both halves of the
    /// exact 128-bit board key go in, so that distinct boards give unrelated terms.
    #[inline]
    const fn sig_terms(h: u64, h2: u64) -> (u64, u64) {
        if Board::<ROWS, COLS, LANES, NPOINTS>::WIDE {
            (mix(h ^ mix2(h2 ^ 0x2545_F491_4F6C_DD1D)), mix2(h2 ^ mix(h ^ 0x9FB2_1C65_1E98_DF25)))
        } else {
            (mix(h), mix2(h))
        }
    }

    #[inline]
    const fn key_of(&self, canon: u64) -> u64 {
        let stm = match self.pos.to_move() {
            Color::Black => STM_BLACK,
            Color::White => STM_WHITE,
        };
        let passes = if self.pos.consecutive_passes() > 0 { PASS_KEY } else { 0 };
        canon ^ stm ^ passes
    }

    /// Play `mv` (already known legal or not) updating path bookkeeping.
    /// Returns the saved state for [`Self::undo`], or the play error.
    #[allow(clippy::type_complexity)]
    fn do_move(
        &mut self,
        mv: Move<ROWS, COLS>,
    ) -> Result<(sanko_core::UndoInfo<ROWS, COLS, LANES, NPOINTS>, [u64; 8], [u64; 8], (u64, u64), u8), PlayError> {
        let mover = self.pos.to_move();
        let undo = self.pos.play(mv, &mut self.hist)?;
        let saved = (self.sym_hashes, self.sym_hashes2, (self.sig, self.sig2), self.path_sym_mask);
        if let Move::Stone(p) = mv {
            let opp = mover.opponent();
            let pi = p.index();
            for s in 0..self.nsym {
                let mut h = self.sym_hashes[s] ^ Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist(s, pi, mover);
                for q in &undo.captures {
                    h ^= Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist(s, q, opp);
                }
                self.sym_hashes[s] = h;
            }
            debug_assert_eq!(self.sym_hashes[0], self.pos.board().hash());
            if Board::<ROWS, COLS, LANES, NPOINTS>::WIDE {
                for s in 0..self.nsym {
                    let mut h = self.sym_hashes2[s] ^ Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist2(s, pi, mover);
                    for q in &undo.captures {
                        h ^= Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist2(s, q, opp);
                    }
                    self.sym_hashes2[s] = h;
                }
                debug_assert_eq!(self.sym_hashes2[0], self.pos.board().hash2());
            }
            let (t1, t2) = Self::sig_terms(self.sym_hashes[0], self.sym_hashes2[0]);
            self.sig ^= t1;
            self.sig2 ^= t2;
            let (canon, _) = self.canon();
            let b = self.pos.board();
            let mut cap = self.path[self.path.len() - 1].cap;
            cap[opp as usize] |= undo.captures.bits();
            self.path.push(PathEntry {
                canon,
                stones: b.stone_count() as u8,
                black: b.stones(Color::Black).bits(),
                white: b.stones(Color::White).bits(),
                cap,
            });
            self.path_sym_mask &= self.stabilizer();
            let ply = self.hist.len() as u64;
            if ply > self.stats.max_ply {
                self.stats.max_ply = ply;
            }
        }
        Ok((undo, saved.0, saved.1, saved.2, saved.3))
    }

    fn undo(
        &mut self,
        mv: Move<ROWS, COLS>,
        st: &(sanko_core::UndoInfo<ROWS, COLS, LANES, NPOINTS>, [u64; 8], [u64; 8], (u64, u64), u8),
    ) {
        self.pos.unmake(&st.0, &mut self.hist);
        self.sym_hashes = st.1;
        self.sym_hashes2 = st.2;
        (self.sig, self.sig2) = st.3;
        self.path_sym_mask = st.4;
        if matches!(mv, Move::Stone(_)) {
            self.path.pop();
        }
    }

    /// Independent second 64-bit hash of the canonical position (canonicalizing
    /// symmetry `sym`), used to verify proof entries on 128 bits.
    fn key2(&self, sym: u8) -> u64 {
        let b = self.pos.board();
        let t = &self.zob2[sym as usize];
        let mut h = match self.pos.to_move() {
            Color::Black => K2_STM_BLACK,
            Color::White => K2_STM_WHITE,
        };
        if self.pos.consecutive_passes() > 0 {
            h ^= K2_PASS;
        }
        for i in &b.stones(Color::Black) {
            h ^= t[i][0];
        }
        for i in &b.stones(Color::White) {
            h ^= t[i][1];
        }
        h
    }

    /// Map a ≤64-point bitboard through symmetry `s`.
    #[inline]
    fn xform(&self, v: u64, s: usize) -> u64 {
        if s == 0 {
            return v;
        }
        let map = &self.sym_map[s];
        let mut out = 0u64;
        let mut x = v;
        while x != 0 {
            let i = x.trailing_zeros() as usize;
            x &= x - 1;
            out |= 1u64 << map[i];
        }
        out
    }

    /// The entry's common-stone sets in the current frame (`sym` = current
    /// canonicalizing symmetry).
    fn common_of(&self, e: &PnEntry, sym: u8) -> (u64, u64) {
        if e.cb == 0 && e.cw == 0 {
            return (0, 0);
        }
        let inv = sym_inverse(sym as usize);
        (self.xform(e.cb, inv), self.xform(e.cw, inv))
    }

    /// May a proven entry `e` for the *current* position (just reached, so its
    /// board is the top of the path) be reused on the current path?
    /// Exact data only: no hash has to be collision-free (see module docs). A
    /// proof proved on this very path takes the same test.
    fn transfers(&mut self, e: &PnEntry, sym: u8) -> bool {
        if self.cfg.exact_only || e.many {
            self.stats.reject_target += 1;
            return false;
        }
        if e.terr != 0 {
            // The proof's static leaves certify the winner's mop-up only on paths
            // where no winner stone was ever captured inside that territory.
            let stm = self.pos.to_move();
            let winner = if e.status == PnStatus::Win { stm } else { stm.opponent() };
            let terr = self.xform(e.terr, sym_inverse(sym as usize));
            if self.path[self.path.len() - 1].cap[winner as usize] & terr != 0 {
                self.stats.reject_terr += 1;
                return false;
            }
        }
        let (ab, aw) = self.common_of(e, sym);
        let n = self.path.len() - 1;
        for pe in &self.path[..n] {
            // An ancestor can coincide with a proof board only if it holds every
            // stone common to all proof boards, has enough stones, and hits the
            // Bloom filter.
            if pe.black & ab == ab
                && pe.white & aw == aw
                && pe.stones >= e.min_stones
                && {
                    // The entry is not path-specific, so `sig` holds the second word.
                    let (b1, b2) = bloom_pair(pe.canon);
                    if b1 == 0 { e.sig & b2 == b2 } else { e.bloom & b1 == b1 }
                }
            {
                self.stats.reject_bloom += 1;
                return false;
            }
        }
        self.stats.reuse_checked += 1;
        true
    }

    fn attrs_of(&self, e: &PnEntry, sym: u8) -> Attrs {
        let (ab, aw) = self.common_of(e, sym);
        Attrs {
            bloom: e.bloom,
            bloom2: if e.many { 0 } else { e.sig },
            target: if e.many { Target::Many(MANY_TABLE) } else { Target::None },
            min_stones: e.min_stones,
            ab,
            aw,
            terr: self.xform(e.terr, sym_inverse(sym as usize)),
        }
    }

    /// Static evaluation of the current position (canonical `key`): `None` if
    /// undecided, else `(winner, Some(territory))` when the pure-eye certificate is
    /// superko-safe on this path (see [`Self::static_leaf_terr`]) or
    /// `(winner, None)` when it is not. Only undecided verdicts are cached.
    fn static_eval(&mut self, key: u64) -> Option<(Color, Option<u64>)> {
        if self.cfg.no_static {
            return None;
        }
        let k2 = self.cfg.komi_times_two;
        if !sanko_core::may_be_decided(&self.pos, k2)
            || self.sc.is_undecided(key)
        {
            return None;
        }
        let (w, life) = sanko_core::statically_decided_life(&self.pos, k2);
        if w.is_none()
            && self.cfg.region_solver
            && !life.black_alive.is_empty()
            && let Some(terr) = self.regions_won(&life)
        {
            let n = NPOINTS as i32;
            if 4 * n > 2 * n + k2 {
                let cap = self.path[self.path.len() - 1].cap[Color::Black as usize];
                self.stats.region_leaves += 1;
                return Some((Color::Black, (cap & terr == 0 && self.game_continues_for(Color::Black)).then_some(terr)));
            }
        }
        let Some(w) = w else {
            // Also after a failed region attempt: `regions_won` depends only on
            // the board and the configuration (its memo holds path-free results),
            // so a later visit would fail again. A region win is not cached.
            self.sc.mark_undecided(key);
            return None;
        };
        let terr = match w {
            Color::Black => life.black_territory.bits(),
            Color::White => life.white_territory.bits(),
        };
        let cap = self.path[self.path.len() - 1].cap[w as usize];
        Some((w, (cap & terr == 0 && self.game_continues_for(w)).then_some(terr)))
    }

    /// Closed-region analysis for Black (the side that must own everything).
    /// Applies when every region (component of non-Black points) is enclosed by
    /// alive Black chains (pure-eye analysis), each of which has two *eyes* (below).
    /// Each other region is then a local game in which White moves first (and
    /// may pass) and Black may only make *permanent* moves: empty points
    /// adjacent to the walls or to earlier fills, which join an unconditionally
    /// alive chain. Black wins a region on reaching a state with no White
    /// stones and every empty point adjacent to Black: the pure-eye analysis then certifies the
    /// whole board once every region is there (eyes keep every chain alive,
    /// and each region is territory of its alive borders).
    ///
    /// Soundness: (i) Composition: Black answers each White move in the region
    /// it was played in and, after a White pass, moves in an unfinished region.
    /// A White move into a region already won (no White stone, every point a
    /// Black stone or adjacent to one) is answered there by the pure-eye
    /// strategy, whose argument is local to the region (the walls are
    /// permanent): White can neither make an eye nor capture there.
    /// (ii) Termination: every Black move adds a permanent stone. (iii)
    /// Positional superko: each Black move adds a permanent stone, so it never
    /// repeats a board of the continuation, and it cannot repeat an earlier
    /// board unless a Black stone was once captured inside a region (the caller
    /// checks the returned territory against the path's captures, exactly as
    /// for pure-eye leaves). (iv) Superko only ever removes White options. The
    /// restriction to permanent moves makes each local game acyclic, hence
    /// history-free and cacheable. It is conservative: kills that need an
    /// interior (vital-point) move are missed.
    ///
    /// Returns the union of the regions (the leaf's territory) if Black wins all.
    fn regions_won(&mut self, life: &sanko_core::PureEyeAnalysis<LANES>) -> Option<u64> {
        let b = self.pos.board();
        let black = b.stones(Color::Black).bits();
        let white = b.stones(Color::White).bits();
        let alive = life.black_alive.bits();
        let full = if NPOINTS >= 64 { u64::MAX } else { (1u64 << NPOINTS) - 1 };
        let mut rest = full & !black;
        let mut terr = 0u64;
        let mut regions: Vec<u64> = Vec::new();
        while rest != 0 {
            let first = rest.trailing_zeros() as usize;
            let region = Self::flood(1u64 << first, rest);
            rest &= !region;
            // Every Black stone bordering the region must be unconditionally alive.
            if Self::dil(region) & black & !alive != 0 {
                return None;
            }
            if region.count_ones() > self.cfg.region_max {
                self.stats.reg_fail[1] += 1;
                return None;
            }
            regions.push(region);
            terr |= region;
        }
        // Every Black chain keeps two *eyes* (regions all of whose points, White
        // stones included, are adjacent to it), vital throughout the
        // continuation: Black only plays there to capture, freeing points that
        // are again adjacent to the chain, and White can never fill an eye's last
        // empty point (no suicide, and the chain cannot be captured while its
        // other eye stands). So each chain stays unconditionally alive while the
        // other regions are fought over (fills merge into it). Implied by the
        // check above (every chain borders a region, so every chain is in X*,
        // which needs two pure eyes among these regions) and kept as a defensive
        // check: its failure counter is 0 in every record run.
        let mut chains = black;
        while chains != 0 {
            let c = Self::flood(1u64 << chains.trailing_zeros(), black);
            chains &= !c;
            let near = Self::dil(c);
            let eyes = regions.iter().filter(|&&r| r & !near == 0).count();
            if eyes < 2 {
                self.stats.reg_fail[2] += 1;
                return None;
            }
        }
        let anchors = Self::dil(black);
        for &region in &regions {
            let a0 = anchors & region;
            if white & region == 0 && region & !a0 == 0 {
                continue; // already won: no White stone, every point anchored
            }
            if !self.local_black_wins(region, a0, white & region, 0, true) {
                self.stats.reg_fail[3] += 1;
                return None;
            }
        }
        Some(terr)
    }

    #[inline]
    const fn dil(b: u64) -> u64 {
        sanko_core::dilate_bits::<COLS, NPOINTS>(b)
    }

    /// Connected component of `seed` within `within`.
    #[inline]
    const fn flood(seed: u64, within: u64) -> u64 {
        let mut comp = seed & within;
        loop {
            let g = Self::dil(comp) & within;
            if g == comp {
                return comp;
            }
            comp = g;
        }
    }

    /// The local game of `regions_won`: `a0` = region points adjacent to the
    /// wall, `white`/`fills` = White stones and Black's permanent fills in the
    /// region. Memoized (the game is acyclic, so results are path-free).
    fn local_black_wins(&mut self, region: u64, a0: u64, white: u64, fills: u64, white_to_move: bool) -> bool {
        if region.count_ones() <= self.cfg.region_interior {
            return self.local_black_wins_interior(region, a0, white, fills, white_to_move);
        }
        let empty = region & !white & !fills;
        let anchored = a0 | Self::dil(fills);
        if white == 0 && empty & !anchored == 0 {
            return true;
        }
        let key = RegionKey { region, a0, white, fills, white_to_move };
        if let Some(&r) = self.region_cache.get(&key) {
            return r;
        }
        let result = if white_to_move {
            // Pass, or any point where the White chain keeps a liberty (White
            // can never capture: every Black stone here is alive).
            let mut ok = self.local_black_wins(region, a0, white, fills, false);
            let mut cand = empty;
            while ok && cand != 0 {
                let p = 1u64 << cand.trailing_zeros();
                cand &= cand - 1;
                let w2 = white | p;
                let chain = Self::flood(p, w2);
                if Self::dil(chain) & empty & !p == 0 {
                    continue; // suicide
                }
                ok = self.local_black_wins(region, a0, w2, fills, false);
            }
            ok
        } else {
            let mut cand = empty & anchored;
            let mut win = false;
            while !win && cand != 0 {
                let p = 1u64 << cand.trailing_zeros();
                cand &= cand - 1;
                let f2 = fills | p;
                let empty2 = empty & !p;
                // Capture White chains left without liberties.
                let mut w2 = white;
                let mut adj = Self::dil(p) & white;
                while adj != 0 {
                    let q = 1u64 << adj.trailing_zeros();
                    let chain = Self::flood(q, w2);
                    adj &= !chain;
                    if Self::dil(chain) & empty2 == 0 {
                        w2 &= !chain;
                    }
                }
                win = self.local_black_wins(region, a0, w2, f2, true);
            }
            win
        };
        // Per-thread memo, bounded: clearing loses only memoized results.
        if self.region_cache.len() > (1 << 20) {
            self.region_cache.clear();
        }
        self.region_cache.insert(key, result);
        result
    }

    /// [`Self::local_black_wins`] with interior moves (`region_interior`).
    /// Black may play any empty point of the region (not suicide). Black stones
    /// connected to the walls (through fills on wall-adjacent points `a0`) are
    /// alive. The others are *interior* and vulnerable: if a White move would
    /// capture any of them, that White move wins the local game for White
    /// (conservative). So in every line Black wins, no Black stone is ever
    /// captured: Black's stones only accumulate, the game stays acyclic and
    /// history-free (memoized on the full state), and each Black move adds a
    /// permanent stone (superko as in `regions_won`). A region is won when it
    /// holds no White stone, all its Black stones are connected to the walls,
    /// and every empty point touches them (or the walls): the same won state as
    /// without interior moves, so the composition argument is unchanged.
    fn local_black_wins_interior(&mut self, region: u64, a0: u64, white: u64, fills: u64, white_to_move: bool) -> bool {
        let empty = region & !white & !fills;
        let connected = if fills & a0 != 0 { Self::flood(fills & a0, fills) } else { 0 };
        let anchored = a0 | Self::dil(connected);
        if white == 0 && fills & !connected == 0 && empty & !anchored == 0 {
            return true;
        }
        let key = RegionKey { region, a0, white, fills, white_to_move };
        if let Some(&r) = self.region_cache.get(&key) {
            return r;
        }
        let interior = fills & !connected;
        let result = if white_to_move {
            // White passes or plays any empty point. A White move that takes the
            // last liberty of an interior Black chain wins for White. Otherwise
            // White's own chain needs a liberty (no suicide).
            let mut ok = self.local_black_wins_interior(region, a0, white, fills, false);
            let mut cand = empty;
            while ok && cand != 0 {
                let p = 1u64 << cand.trailing_zeros();
                cand &= cand - 1;
                let empty2 = empty & !p;
                let mut captures_black = false;
                let mut adj = Self::dil(p) & interior;
                while adj != 0 {
                    let q = 1u64 << adj.trailing_zeros();
                    let chain = Self::flood(q, interior);
                    adj &= !chain;
                    if Self::dil(chain) & empty2 == 0 {
                        captures_black = true;
                        break;
                    }
                }
                if captures_black {
                    ok = false;
                    break;
                }
                let w2 = white | p;
                let chain = Self::flood(p, w2);
                if Self::dil(chain) & empty2 == 0 {
                    continue; // suicide
                }
                ok = self.local_black_wins_interior(region, a0, w2, fills, false);
            }
            ok
        } else {
            let mut cand = empty;
            let mut win = false;
            while !win && cand != 0 {
                let p = 1u64 << cand.trailing_zeros();
                cand &= cand - 1;
                let f2 = fills | p;
                let empty2 = empty & !p;
                // Capture White chains left without liberties.
                let mut w2 = white;
                let mut adj = Self::dil(p) & white;
                while adj != 0 {
                    let q = 1u64 << adj.trailing_zeros();
                    let chain = Self::flood(q, w2);
                    adj &= !chain;
                    if Self::dil(chain) & empty2 == 0 {
                        w2 &= !chain;
                    }
                }
                // An interior Black chain must keep a liberty (no suicide),
                // while a chain reaching the walls is alive.
                let chain = Self::flood(p, f2);
                if chain & a0 == 0 {
                    let empty3 = region & !w2 & !f2;
                    if Self::dil(chain) & empty3 == 0 {
                        continue;
                    }
                }
                win = self.local_black_wins_interior(region, a0, w2, f2, true);
            }
            win
        };
        // Per-thread memo, bounded: clearing loses only memoized results.
        if self.region_cache.len() > (1 << 20) {
            self.region_cache.clear();
        }
        self.region_cache.insert(key, result);
        result
    }

    /// Pure-eye certificate check under positional superko. The static verdict
    /// says `w` wins by mopping up inside its unconditional territory: each of
    /// those moves adds a permanent `w` stone there, so it can repeat an earlier
    /// board only if some earlier board had a `w` stone inside the territory,
    /// i.e. one was captured there on this path. Returns the territory (the
    /// leaf's `terr` attribute) if the certificate is safe on this path.
    fn static_leaf_terr(&self, w: Color) -> Option<u64> {
        let life = sanko_core::pure_eye_analysis(self.pos.board());
        let terr = match w {
            Color::Black => life.black_territory.bits(),
            Color::White => life.white_territory.bits(),
        };
        let cap = self.path[self.path.len() - 1].cap[w as usize];
        (cap & terr == 0 && self.game_continues_for(w)).then_some(terr)
    }

    /// Whether a static (mop-up) win for `w` can be realized from here: not if
    /// `w` just passed and the opponent is to move. The opponent may pass too,
    /// ending the game on the *current* area count (with `w`'s territory still
    /// holding opponent stones). Found by the no-static oracle (3×3 selftest).
    fn game_continues_for(&self, w: Color) -> bool {
        !(self.pos.consecutive_passes() == 1 && self.pos.to_move() != w)
    }

    /// Classify the current position (a freshly generated child): terminal,
    /// static, TT proof, or unproven with stored/initial numbers.
    fn classify(&mut self, mv_code: u8) -> Child {
        self.stats.children += 1;
        let (canon, sym) = self.canon();
        let key = self.key_of(canon);
        // A leaf contributes its own board only when it is a loss for its side
        // to move reached by a stone move: then that board was produced by the
        // proof's *winner* (see `Attrs`).
        let produced = if self.pos.consecutive_passes() == 0 {
            self.board_attrs()
        } else {
            Attrs::empty::<NPOINTS>()
        };
        let stm = self.pos.to_move();
        let (phi0, delta0) = if self.cfg.prior_weights.is_some() {
            // Set at the end, only for a child that keeps its initial numbers.
            (1, 1)
        } else if self.cfg.eye_init > 0 {
            // Kill-everything prior: White's potential eye points are empties
            // with no Black neighbour. Killing costs ~1 + E, living ~1 + (T - E).
            let b = self.pos.board();
            let near_black = sanko_core::dilate4::<ROWS, COLS, LANES, NPOINTS>(&b.stones(Color::Black));
            let space = if self.cfg.eye_area {
                // Empties and White stones alike: White's black-free area.
                b.empties().union(&b.stones(Color::White))
            } else {
                b.empties()
            };
            let e = u64::from(space.difference(&near_black).count());
            // Optionally weight White's existing eye points (empties whose every
            // neighbour is White) as extra area.
            let e = if self.cfg.eye_weight > 0 {
                // An eye point: empty, and no neighbour is empty or Black.
                let empt = b.empties().bits();
                let open = empt | b.stones(Color::Black).bits();
                let eyes = empt & !sanko_core::neighbors_bits::<COLS, NPOINTS>(open);
                e + u64::from(eyes.count_ones()) * u64::from(self.cfg.eye_weight)
            } else {
                e
            };
            // With `eye_komi`, White's area counts only beyond what Black can
            // concede: Black wins iff White's area w (no dame) has
            // NPOINTS - 2w > komi, i.e. w <= ceil((2N - komi2) / 4) - 1.
            let e = if self.cfg.eye_komi {
                let allowed = (2 * NPOINTS as i64 - i64::from(self.cfg.komi_times_two)).max(0) as u64;
                e.saturating_sub(allowed.div_ceil(4).saturating_sub(1))
            } else {
                e
            };
            let kill = 1 + e;
            let live = 1 + u64::from(self.cfg.eye_init).saturating_sub(e);
            if stm == Color::Black { (kill, live) } else { (live, kill) }
        } else {
            (1, 1)
        };
        let mut c = Child {
            mv: mv_code,
            phi: phi0,
            delta: delta0,
            key,
            base_key: key,
            twin: false,
            attrs: Attrs::UNPROVEN,
            fresh: false,
        };
        let solved = |c: &mut Child, stm_wins: bool, attrs: Attrs| {
            debug_assert!(attrs != Attrs::UNPROVEN, "a solved node needs real proof attributes");
            if stm_wins {
                c.phi = 0;
                c.delta = INF;
            } else {
                c.phi = INF;
                c.delta = 0;
            }
            c.attrs = attrs;
        };
        if self.pos.is_terminal() {
            self.stats.terminal_leaves += 1;
            let w = terminal_winner(&self.pos, self.cfg.komi_times_two).unwrap_or(Color::White);
            solved(&mut c, w == stm, Attrs::empty::<NPOINTS>());
            return c;
        }
        if let Some((w, terr)) = self.static_eval(key) {
            if let Some(terr) = terr {
                self.stats.static_leaves += 1;
                let mut a = if w == stm { Attrs::empty::<NPOINTS>() } else { produced };
                a.terr = terr;
                solved(&mut c, w == stm, a);
                return c;
            }
            self.stats.static_rejects += 1;
        }
        if self.pos.consecutive_passes() == 1
            && terminal_winner(&self.pos, self.cfg.komi_times_two).unwrap_or(Color::White) == stm
        {
            // The opponent just passed: the side to move wins by passing back
            // (always legal, and it ends the game on this very board).
            self.stats.terminal_leaves += 1;
            solved(&mut c, true, Attrs::empty::<NPOINTS>());
            return c;
        }
        self.stats.tt_probes += 1;
        match self.tt.probe(key) {
            Some(e) if e.status != PnStatus::Unproven => {
                self.stats.tt_hits += 1;
                let k2 = self.key2(sym);
                if e.check != k2 {
                    self.stats.check_mismatch += 1;
                }
                if e.check == k2 && self.transfers(&e, sym) {
                    let a = self.attrs_of(&e, sym);
                    solved(&mut c, e.status == PnStatus::Win, a);
                    return c;
                }
                for alt in ALT_KEYS {
                    self.stats.tt_probes += 1;
                    if let Some(t) = self.tt.probe(key ^ alt)
                        && t.status != PnStatus::Unproven
                        && t.check == k2
                        && self.transfers(&t, sym)
                    {
                        self.stats.alt_hits += 1;
                        let a = self.attrs_of(&t, sym);
                        solved(&mut c, t.status == PnStatus::Win, a);
                        return c;
                    }
                }
                let tkey = key ^ self.sig;
                c.key = tkey;
                c.twin = true;
                self.stats.tt_probes += 1;
                if let Some(t) = self.tt.probe(tkey) {
                    self.stats.tt_hits += 1;
                    if t.status == PnStatus::Unproven {
                        c.phi = from_tt(t.phi);
                        c.delta = from_tt(t.delta);
                        return c;
                    } else if t.check != k2 ^ self.sig2 {
                        self.stats.check_mismatch += 1;
                    } else if t.many && t.sym == sym && t.sig == self.sig {
                        // Same canonical key, path signature and frame: the
                        // same raw board with the same set of earlier boards,
                        // i.e. the identical superko game. (A mirror image of
                        // the board with the same board set is another game.
                        // A path whose signature equals an `ALT_KEYS` constant
                        // has an alternate slot as its twin key: the entry
                        // there is not path-specific, `many` rejects it.)
                        self.stats.twin_hits += 1;
                        let a = self.attrs_of(&t, sym);
                        solved(&mut c, t.status == PnStatus::Win, a);
                    } else {
                        self.stats.twin_frame_rejects += 1;
                    }
                }
            }
            Some(e) => {
                self.stats.tt_hits += 1;
                c.phi = from_tt(e.phi);
                c.delta = from_tt(e.delta);
                if e.twinned {
                    self.stats.tt_probes += 1;
                    if let Some(t) = self.tt.probe(key ^ self.sig)
                        && t.status != PnStatus::Unproven
                        && t.check == self.key2(sym) ^ self.sig2
                    {
                        if t.many && t.sym == sym && t.sig == self.sig {
                            self.stats.twin_hits += 1;
                            let a = self.attrs_of(&t, sym);
                            solved(&mut c, t.status == PnStatus::Win, a);
                        } else {
                            self.stats.twin_frame_rejects += 1;
                        }
                    }
                }
                // Table numbers, not the prior.
                return c;
            }
            None => c.fresh = true,
        }
        if let Some(w) = self.cfg.prior_weights
            && c.phi != 0
            && c.delta != 0
        {
            (c.phi, c.delta) = self.learned_prior(&w, stm);
        }
        c
    }

    /// Initial numbers from the learned prior: `1 + scale · (−ln p)` for the
    /// proof number, with `p` the model's probability that `stm` wins, and the
    /// same with `1 − p` for the disproof number. Always in `[1, 2^40]`, whatever
    /// the weights and scale (NaN casts to 0): an unsolved child must never get
    /// 0 or the solved value.
    fn learned_prior(&self, weights: &[f32; PRIOR_FEATURES], stm: Color) -> (u64, u64) {
        let logit: f32 = weights.iter().zip(self.prior_features().iter()).map(|(a, b)| a * b).sum();
        let p_black = 1.0 / (1.0 + (-logit).exp());
        let win = if stm == Color::Black { p_black } else { 1.0 - p_black }.clamp(1e-4, 1.0 - 1e-4);
        let scale = self.cfg.prior_scale;
        let number = |x: f32| (x.round() as u64).clamp(1, 1 << 40);
        (number(scale.mul_add(-win.ln(), 1.0)), number(scale.mul_add(-(1.0 - win).ln(), 1.0)))
    }

    /// Features of the current position for the learned prior, from Black's
    /// side: bias, Black to move, komi, stones of each colour, White's black-free
    /// area (the eye-space prior), eye points of each colour, chains in atari and
    /// with two liberties of each colour, and liberties of each colour. Counts
    /// are scaled by the number of points (or by 4 for chain counts).
    fn prior_features(&self) -> [f32; PRIOR_FEATURES] {
        let b = self.pos.board();
        let n = NPOINTS as f32;
        let black = b.stones(Color::Black).bits();
        let white = b.stones(Color::White).bits();
        let empt = b.empties().bits();
        let (ba, b2, bl) = chain_stats::<COLS, NPOINTS>(black, empt);
        let (wa, w2, wl) = chain_stats::<COLS, NPOINTS>(white, empt);
        let near_black = sanko_core::dilate_bits::<COLS, NPOINTS>(black);
        let space = (empt | white) & !near_black;
        let white_eyes = empt & !sanko_core::neighbors_bits::<COLS, NPOINTS>(empt | black);
        let black_eyes = empt & !sanko_core::neighbors_bits::<COLS, NPOINTS>(empt | white);
        let c = |x: u32| x as f32 / n;
        [
            1.0,
            f32::from(u8::from(self.pos.to_move() == Color::Black)),
            self.cfg.komi_times_two as f32 / (2.0 * n),
            c(black.count_ones()),
            c(white.count_ones()),
            c(space.count_ones()),
            c(white_eyes.count_ones()),
            c(black_eyes.count_ones()),
            ba as f32 / 4.0,
            wa as f32 / 4.0,
            b2 as f32 / 4.0,
            w2 as f32 / 4.0,
            c(bl),
            c(wl),
        ]
    }

    /// Is `p` a single-point eye of `color` (all on-board neighbours are its
    /// stones and at most one diagonal, or none on the edge, is hostile)?
    fn is_own_eye(&self, p: Point<ROWS, COLS>, color: Color) -> bool {
        let board = self.pos.board();
        for n in p.neighbors().into_iter().flatten() {
            if board.at(n) != Some(color) {
                return false;
            }
        }
        let (r, c) = (p.row() as isize, p.col() as isize);
        let mut bad = 0;
        let mut edge = false;
        for (dr, dc) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            if rr < 0 || cc < 0 || rr >= ROWS as isize || cc >= COLS as isize {
                edge = true;
                continue;
            }
            let q = Point::<ROWS, COLS>::new_unchecked(rr as usize, cc as usize);
            if board.at(q) == Some(color.opponent()) {
                bad += 1;
            }
        }
        if edge { bad == 0 } else { bad <= 1 }
    }

    /// Generate and classify all children of the current node into `kids`.
    /// Returns `(winning child index if one is already proven to lose for the
    /// opponent, superko exclusion targets, whether a symmetry merge occurred)`.
    fn generate(&mut self, kids: &mut Vec<Child>) -> (Option<usize>, Target, bool) {
        kids.clear();
        let stm = self.pos.to_move();
        let empties = self.pos.board().empties();
        if self.cfg.prefetch {
            self.prefetch_children(stm, empties.bits());
        }
        let mut excl = Target::None;
        let mut merged = false;
        let prune_eyes = self.cfg.prune_for == Some(stm);
        // The pruned side also never plays inside its own certified territory: the
        // position becomes a static leaf once every region is secure, so such
        // moves are never needed, except where the static rule refuses (a stone
        // of ours was captured there on this path), in which case they are kept.
        let own_terr = if prune_eyes && !self.cfg.no_static {
            // Only the side to move's analysis is needed.
            let t = sanko_core::pure_eye_analysis_of(self.pos.board(), stm).1.bits();
            if self.path[self.path.len() - 1].cap[stm as usize] & t == 0 { t } else { 0 }
        } else {
            0
        };
        let mask = self.path_sym_mask & !1;
        let mut tempo_kept = false;
        for k in 0..=NPOINTS {
            let mut is_tempo = false;
            let mv = if k == NPOINTS {
                Move::Pass
            } else {
                let idx = self.order[k];
                if !empties.get(idx) {
                    continue;
                }
                if mask != 0 {
                    let mut dup = false;
                    for s in 1..self.nsym {
                        if mask & (1 << s) != 0 && self.rank[self.sym_map[s][idx]] < k {
                            dup = true;
                            break;
                        }
                    }
                    if dup {
                        merged = true;
                        self.stats.sym_merged += 1;
                        continue;
                    }
                }
                let p = Point::<ROWS, COLS>::from_index_unchecked(idx);
                if prune_eyes && ((own_terr >> idx) & 1 != 0 || self.is_own_eye(p, stm)) {
                    if self.cfg.keep_tempo && !tempo_kept {
                        tempo_kept = true;
                        is_tempo = true;
                    } else {
                        self.stats.terr_pruned += 1;
                        continue;
                    }
                }
                Move::Stone(p)
            };
            let st = match self.do_move(mv) {
                Ok(st) => st,
                Err(e) => {
                    // an illegal move does not use up the kept tempo move
                    tempo_kept &= !is_tempo;
                    if e == PlayError::PositionalSuperko {
                        self.stats.psk_exclusions += 1;
                        if let Move::Stone(p) = mv
                            && let Some(h) = self.pos.board_hash_after(p)
                        {
                            excl = excl.union(Target::One(h));
                        }
                    }
                    continue;
                }
            };
            let c = self.classify(mv.encode() as u8);
            self.undo(mv, &st);
            kids.push(c);
            if c.delta == 0 && !self.list_all {
                return (Some(kids.len() - 1), excl, merged);
            }
        }
        (None, excl, merged)
    }

    /// Every child of the current node as the search generates it (moves
    /// merged under the symmetries of the whole game so far, the pruned side's
    /// skipped moves left out), with its numbers `(move code, phi, delta)` seen
    /// from the position after the move: delta 0 marks a winning move, phi 0 a
    /// refuted one. A stored proof counts only where it transfers to this path.
    fn children_numbers(&mut self) -> Vec<(usize, u64, u64)> {
        self.list_all = true;
        let mut kids = Vec::new();
        self.generate(&mut kids);
        self.list_all = false;
        kids.iter().map(|k| (usize::from(k.mv), k.phi, k.delta)).collect()
    }

    /// Load the TT bucket and static-cache slot of every stone child (computed
    /// without playing the move) so the later probes hit warm lines. The
    /// independent loads overlap instead of missing one after another.
    fn prefetch_children(&self, stm: Color, empties: u64) {
        let opp = stm.opponent();
        let child_stm = match opp {
            Color::Black => STM_BLACK,
            Color::White => STM_WHITE,
        };
        let mut cand = empties;
        while cand != 0 {
            let idx = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            let p = Point::<ROWS, COLS>::from_index_unchecked(idx);
            let Some((caps, _)) = self.pos.preview_stone(p) else { continue };
            let mut canon = u64::MAX;
            for s in 0..self.nsym {
                let mut h = self.sym_hashes[s] ^ Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist(s, idx, stm);
                for q in &caps {
                    h ^= Board::<ROWS, COLS, LANES, NPOINTS>::sym_zobrist(s, q, opp);
                }
                canon = canon.min(h);
            }
            let key = canon ^ child_stm;
            self.tt.touch(key);
            self.sc.touch(key);
        }
    }

    /// The current board as a one-board attribute set.
    fn board_attrs(&self) -> Attrs {
        let top = self.path[self.path.len() - 1];
        Attrs {
            bloom: bloom_pair(top.canon).0,
            bloom2: bloom_pair(top.canon).1,
            target: Target::None,
            min_stones: top.stones,
            ab: top.black,
            aw: top.white,
            terr: 0,
        }
    }

    /// Attributes of a node won through a child with attributes `a`: the move
    /// into this node was the loser's, so only the internal-exclusion
    /// bookkeeping changes.
    const fn win_attrs(&self, a: Attrs) -> Attrs {
        Attrs {
            target: a.target.without(self.sym_hashes[0]),
            ..a
        }
    }

    /// Attributes of a lost node from the union `a` of its children's: its own
    /// board was produced by the winner's move into it (unless that was a pass).
    fn loss_attrs(&self, a: Attrs) -> Attrs {
        let a = if self.pos.consecutive_passes() == 0 { a.join(self.board_attrs()) } else { a };
        Attrs {
            target: a.target.without(self.sym_hashes[0]),
            ..a
        }
    }

    fn store_proof(&mut self, c: &Child, win: bool, attrs: Attrs, pv_code: u8, work: u64) {
        debug_assert!(attrs != Attrs::UNPROVEN, "a proof needs real attributes");
        let (_, sym) = self.canon();
        // PV in the canonical frame: apply this node's canonicalizing symmetry.
        let pv = if win && (pv_code as usize) < NPOINTS {
            self.sym_map[sym as usize][pv_code as usize] as u8
        } else {
            pv_code
        };
        let mut e = PnEntry::unproven(c.key, 0, 0, work);
        e.status = if win { PnStatus::Win } else { PnStatus::Loss };
        e.phi = if win { 0 } else { TT_INF };
        e.delta = if win { TT_INF } else { 0 };
        e.bloom = attrs.bloom;
        e.sig = self.sig;
        e.sym = sym;
        e.pv = pv;
        e.min_stones = attrs.min_stones;
        e.cb = self.xform(attrs.ab, sym as usize);
        e.cw = self.xform(attrs.aw, sym as usize);
        e.terr = self.xform(attrs.terr, sym as usize);
        e.check = self.key2(sym);
        e.black = self.pos.to_move() == Color::Black;
        self.stats.proofs_stored += 1;
        if let Some((rate, w)) = PROOF_DUMP.get()
            && c.base_key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40 < (1u64 << 24) / rate
        {
            let stm_wins = win;
            let black_wins = stm_wins == (self.pos.to_move() == Color::Black);
            let f = self.prior_features();
            let mut rec = Vec::with_capacity(4 * (PRIOR_FEATURES + 2));
            for x in f.iter().chain([&f32::from(u8::from(black_wins)), &((work as f32) + 1.0).log2()]) {
                rec.extend_from_slice(&x.to_le_bytes());
            }
            if let Ok(mut w) = w.lock() {
                let _ = std::io::Write::write_all(&mut *w, &rec);
            }
        }
        // The table stores neither exclusion targets nor (narrow layout)
        // territory beyond 32 points: such proofs are path-specific (exact-path
        // reuse only), as are proofs depending on several targets or a
        // symmetry merge.
        if matches!(attrs.target, Target::One(_)) {
            self.stats.proofs_with_target += 1;
        }
        let causes = attrs.target.many_causes();
        for (i, bit) in [MANY_OVERFLOW, MANY_MERGE, MANY_TABLE].into_iter().enumerate() {
            if causes & bit != 0 {
                self.stats.path_specific[i] += 1;
            }
        }
        // The exact-only oracle keeps every proof under its twin key (it reuses
        // proofs only on the identical game, which only the twin key identifies).
        if matches!(attrs.target, Target::One(_) | Target::Many(_))
            || (!self.tt.wide() && e.terr >> 32 != 0)
            || self.cfg.exact_only
        {
            e.key = c.base_key ^ self.sig;
            e.check ^= self.sig2;
            e.many = true;
            self.tt.store(&e);
            if !c.twin {
                self.tt.mark_twinned(c.base_key);
            }
            self.stats.twin_stores += 1;
        } else {
            // A transferable proof always takes the full transfer test, so it
            // needs no path signature: `sig` holds the second Bloom word.
            e.sig = attrs.bloom2;
            if c.twin {
                // A target-free proof of a child searched under its twin key (the
                // base slot holds another path's proof): an alternate base proof.
                self.stats.path_specific[3] += 1;
                self.store_alternate(&mut e, c.base_key);
            } else if !self.tt.store_base_proof(&e) {
                // Another path's proof already holds the base slot.
                self.stats.path_specific[4] += 1;
                self.store_alternate(&mut e, c.base_key);
            }
        }
    }

    /// Store a target-free proof under an alternate base key (see `ALT_KEYS`),
    /// chosen by the path signature so that different paths spread over them.
    /// A later proof from another path may replace it, which costs only time.
    fn store_alternate(&mut self, e: &mut PnEntry, base_key: u64) {
        // The slot is chosen by the path signature (`e.sig` holds the second Bloom word).
        e.key = base_key ^ ALT_KEYS[(self.sig >> 7) as usize % ALT_KEYS.len()];
        e.many = false;
        self.tt.store(e);
        self.stats.alt_stores += 1;
    }

    /// The df-pn MID procedure for the current position (whose TT identity is
    /// `me`). Returns the node's `(phi, delta, attrs)`. With several workers the
    /// node is marked busy for the duration (virtual loss for the others).
    fn mid(&mut self, me: Child, th_phi: u64, th_delta: u64, depth: usize) -> (u64, u64, Attrs) {
        if !self.par {
            return self.mid_inner(me, th_phi, th_delta, depth);
        }
        let slot = self.sh.slot(me.key);
        slot.fetch_add(1, Ordering::Relaxed);
        let r = self.mid_inner(me, th_phi, th_delta, depth);
        slot.fetch_sub(1, Ordering::Relaxed);
        r
    }

    fn mid_inner(&mut self, me: Child, th_phi: u64, th_delta: u64, depth: usize) -> (u64, u64, Attrs) {
        self.stats.nodes += 1;
        if me.fresh {
            self.stats.fresh += 1;
        }
        let work_start = self.stats.nodes;
        self.pending_nodes += 1;
        if self.pending_nodes >= 4096 {
            let total = self.sh.nodes.fetch_add(self.pending_nodes, Ordering::Relaxed) + self.pending_nodes;
            self.pending_nodes = 0;
            if self.sh.pause.load(Ordering::Relaxed) {
                self.sh.park();
            }
            if self.cfg.node_limit != 0 && total >= self.cfg.node_limit {
                self.sh.stop.store(true, Ordering::Relaxed);
            }
            if self.sh.stop.load(Ordering::Relaxed) {
                self.aborted = true;
            }
            if self.tid == 0 && total >= self.next_report {
                self.report(depth, total);
            }
        }
        if self.par && !self.aborted && self.sh.stop.load(Ordering::Relaxed) {
            self.aborted = true;
        }
        if self.aborted || self.hist.len() >= MAX_PLY {
            // Give up on this subtree for now: report "threshold reached". At
            // MAX_PLY the search is not marked aborted, so a line deeper than
            // that can never be proven and the root would keep searching (no
            // wrong result, but no result either). Searches stay far below it:
            // the deepest seen was 200 plies (1×15), about 80 on wider boards.
            return (th_phi.max(1), th_delta.max(1), Attrs::UNPROVEN);
        }
        if self.bufs.len() <= depth {
            self.bufs.push(Vec::with_capacity(NPOINTS + 1));
        }
        let mut kids = std::mem::take(&mut self.bufs[depth]);
        let (win_now, excl, merged) = self.generate(&mut kids);

        let result;
        if let Some(i) = win_now {
            let attrs = self.win_attrs(kids[i].attrs);
            let pv = kids[i].mv;
            self.store_proof(&me, true, attrs, pv, 1);
            result = (0, INF, attrs);
        } else {
            // Simulation: the last child proven a win for the opponent (its
            // stored proof is the source), and the children already tried.
            let mut sim_last = usize::MAX;
            let mut sim_tried = 0u64;
            // `sim_tried` has a bit per child (at most NPOINTS + 1 <= 50).
            let sim_on = self.cfg.simulate > 0 && kids.len() <= 64;
            loop {
                if self.par {
                    // See the other workers' progress on unsolved siblings. Only
                    // their unproven entries are read (a stored proof is not used
                    // here), and their buckets and the busy counters the selection
                    // below reads are prefetched first, so the misses overlap.
                    let vl = self.cfg.virtual_loss;
                    for c in &kids {
                        if c.phi != 0 && c.delta != 0 {
                            self.tt.touch_unproven(c.key);
                            if vl {
                                crate::pntt::prefetch(self.sh.slot(c.key));
                            }
                        }
                    }
                    for c in &mut kids {
                        if c.phi != 0
                            && c.delta != 0
                            && let Some((p, d)) = self.tt.probe_numbers(c.key)
                        {
                            // Only ever raise: TT numbers are rounded down
                            // (15-bit floats), and lowering a value a child
                            // just returned above its threshold would make
                            // the loop re-select it forever.
                            c.phi = c.phi.max(from_tt(p));
                            c.delta = c.delta.max(from_tt(d));
                        }
                    }
                }
                // phi = min delta(c), and delta = Σ phi(c). Selection uses delta
                // inflated by the number of other workers inside the child.
                let mut phi = INF;
                let mut best = usize::MAX;
                let mut best_eff = u64::MAX;
                let mut delta2_eff = u64::MAX;
                let mut sum: u64 = 0;
                for (i, c) in kids.iter().enumerate() {
                    phi = phi.min(c.delta);
                    let eff = if self.par && self.cfg.virtual_loss && c.delta != 0 {
                        let b = u64::from(self.sh.slot(c.key).load(Ordering::Relaxed));
                        if b == 0 {
                            c.delta
                        } else {
                            c.delta.saturating_add(c.delta.saturating_mul(b))
                        }
                    } else {
                        c.delta
                    };
                    // An unproven child must stay selectable however saturated
                    // its virtual-loss inflation (else `best` could stay unset).
                    let eff = if c.delta < INF { eff.min(INF - 1) } else { eff };
                    if eff < best_eff {
                        delta2_eff = best_eff;
                        best_eff = eff;
                        best = i;
                    } else if eff < delta2_eff {
                        delta2_eff = eff;
                    }
                    sum = sum.saturating_add(c.phi);
                }
                if phi == 0 {
                    // Make sure `best` is a proven-losing child.
                    best = kids.iter().position(|c| c.delta == 0).expect("phi == 0");
                }
                let delta2 = delta2_eff;
                if phi == 0 {
                    let attrs = self.win_attrs(kids[best].attrs);
                    let pv = kids[best].mv;
                    let work = self.stats.nodes - work_start + 1;
                    self.store_proof(&me, true, attrs, pv, work);
                    result = (0, INF, attrs);
                    break;
                }
                if sum == 0 {
                    // Every child is a win for the opponent: a loss.
                    let mut a = Attrs { target: excl, ..Attrs::empty::<NPOINTS>() };
                    for c in &kids {
                        a = a.join(c.attrs);
                    }
                    if merged {
                        a.target = Target::Many(MANY_MERGE | a.target.many_causes());
                    }
                    let attrs = self.loss_attrs(a);
                    let work = self.stats.nodes - work_start + 1;
                    self.store_proof(&me, false, attrs, NPOINTS as u8, work);
                    result = (INF, 0, attrs);
                    break;
                }
                let delta = sum.min(MAX_UNSOLVED);
                if depth < 4 {
                    self.path_pd[depth] = (kids[best].mv, phi, delta);
                }
                if depth == 0 {
                    self.root_pd = (phi, delta);
                    if self.cfg.progress && self.tid == 0 {
                        self.root_kids = kids.iter().map(|k| (k.mv, k.phi, k.delta, k.key)).collect();
                        self.root_sym_mask = self.path_sym_mask;
                    }
                }
                if phi >= th_phi || delta >= th_delta || self.aborted {
                    let work = self.stats.nodes - work_start + 1;
                    // The store keeps an existing slot's twinned flag, so this
                    // never clears a flag set by a path-specific proof.
                    let e = PnEntry::unproven(me.key, to_tt(phi), to_tt(delta), work);
                    self.tt.store(&e);
                    result = (phi, delta, Attrs::UNPROVEN);
                    break;
                }
                // Child thresholds (standard df-pn with the 1+ε trick).
                let eps = self.cfg.eps;
                let thresholds = |c: &Child, delta2: u64| -> (u64, u64) {
                    let others = sum.saturating_sub(c.phi);
                    // Normally `others + c.phi = delta < th_delta`, so the child
                    // gets room above its phi. When the sum has saturated (DAG
                    // double counting makes proof numbers count paths, which can
                    // exceed 2^62), that room vanishes and the child would return
                    // at once, forever: then drive progress by the delta threshold
                    // alone.
                    let c_th_phi = if others.saturating_add(c.phi) >= th_delta {
                        INF - 1
                    } else {
                        th_delta - others
                    };
                    let d2p1 = delta2.saturating_add(1);
                    let d2eps = if eps > 0.0 && delta2 < MAX_UNSOLVED {
                        ((delta2 as f64 * (1.0 + eps)).ceil() as u64).min(INF - 1)
                    } else {
                        d2p1
                    };
                    (c_th_phi, th_phi.min(d2p1.max(d2eps)))
                };
                let mut c = kids[best];
                let (mut c_th_phi, mut c_th_delta) = thresholds(&c, delta2);
                if c_th_delta <= c.delta || c_th_phi <= c.phi {
                    // The virtual-loss pick cannot make progress within the
                    // parent's thresholds: fall back to the true most-proving child.
                    // Only reachable with virtual loss (parallel runs). In a serial
                    // run the pick is already the most-proving child, whose
                    // thresholds always exceed its numbers.
                    let mut rb = 0;
                    let mut rd2 = INF;
                    for (i, k) in kids.iter().enumerate() {
                        if k.delta < kids[rb].delta || i == 0 {
                            if i != 0 {
                                rd2 = kids[rb].delta;
                            }
                            rb = i;
                        } else if k.delta < rd2 {
                            rd2 = k.delta;
                        }
                    }
                    best = rb;
                    c = kids[rb];
                    (c_th_phi, c_th_delta) = thresholds(&c, rd2);
                }
                if sim_on && sim_last != usize::MAX && !self.aborted && sim_tried & (1u64 << best) == 0 {
                    sim_tried |= 1u64 << best;
                    if let Some(a) = self.simulate(kids[sim_last].mv, &c) {
                        let k = &mut kids[best];
                        k.phi = 0;
                        k.delta = INF;
                        k.attrs = a;
                        continue;
                    }
                }
                let mv = Move::<ROWS, COLS>::decode(c.mv as usize).expect("valid move code");
                let st = self.do_move(mv).expect("child move was legal at generation");
                let (cphi, cdelta, cattrs) = self.mid(c, c_th_phi, c_th_delta, depth + 1);
                self.undo(mv, &st);
                let k = &mut kids[best];
                k.phi = cphi;
                k.delta = cdelta;
                k.attrs = cattrs;
                if cphi == 0 {
                    // A new source: try every unsolved sibling not tried yet.
                    sim_last = best;
                    if sim_on && !self.aborted {
                        let src = kids[best].mv;
                        for (j, k) in kids.iter_mut().enumerate() {
                            if k.phi != 0 && k.delta != 0 && sim_tried & (1u64 << j) == 0 {
                                sim_tried |= 1u64 << j;
                                if let Some(a) = self.simulate(src, k) {
                                    k.phi = 0;
                                    k.delta = INF;
                                    k.attrs = a;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.bufs[depth] = kids;
        result
    }

    /// Kawano's simulation: try to prove that the opponent wins after the side
    /// to move plays `target`, given that the opponent wins after `src` (its
    /// proof is stored). The shadow position follows `src`'s line: at the
    /// opponent's turns the stored winning move of the shadow position is
    /// played on both boards, and at ours every legal reply is answered on both.
    /// Everything is verified on the real position with the search's own
    /// classification and proof attributes (the shadow only suggests moves), and
    /// every node proven on the way is stored. `None` when a suggestion is
    /// missing or fails, a reply is illegal on the shadow, or the budget runs out.
    fn simulate(&mut self, src: u8, target: &Child) -> Option<Attrs> {
        self.stats.sim_attempts += 1;
        let smv = Move::<ROWS, COLS>::decode(src as usize)?;
        self.shadow_pos.clone_from(&self.pos);
        let su = self.shadow_pos.play(smv, &mut self.shadow_hist).ok()?;
        let ssig = if matches!(smv, Move::Stone(_)) { self.sig ^ Self::sig_terms(self.shadow_pos.board().hash(), self.shadow_pos.board().hash2()).0 } else { self.sig };
        let mv = Move::<ROWS, COLS>::decode(target.mv as usize).expect("valid move code");
        let st = self.do_move(mv).expect("child move was legal at generation");
        let mut budget = self.cfg.simulate;
        let r = self.sim_win(target, ssig, &mut budget, 0);
        self.undo(mv, &st);
        self.shadow_pos.unmake(&su, &mut self.shadow_hist);
        // Simulated nodes are not expansions (`stats.nodes`), but they count
        // towards the node limit and the progress of the stop polling.
        let used = self.cfg.simulate - budget;
        self.stats.sim_nodes += used;
        self.pending_nodes += used;
        if r.is_some() {
            self.stats.sim_successes += 1;
        }
        r
    }

    /// Simulation at a node where the opponent of the original side is to move
    /// (`me`, the current position): play the shadow's suggestion.
    fn sim_win(&mut self, me: &Child, ssig: u64, budget: &mut u64, d: usize) -> Option<Attrs> {
        if *budget == 0 || self.hist.len() >= MAX_PLY {
            return None;
        }
        let start = *budget;
        *budget -= 1;
        let hint = self.shadow_hint(ssig)?;
        let mv = Move::<ROWS, COLS>::decode(hint)?;
        let st = self.do_move(mv).ok()?;
        let r = if let Ok(su) = self.shadow_pos.play(mv, &mut self.shadow_hist) {
            let s2 = if matches!(mv, Move::Stone(_)) { ssig ^ Self::sig_terms(self.shadow_pos.board().hash(), self.shadow_pos.board().hash2()).0 } else { ssig };
            let c = self.classify(hint as u8);
            let r = if c.delta == 0 {
                // A leaf or stored proof: the side to move there loses.
                Some(c.attrs)
            } else if c.phi == 0 {
                None
            } else {
                self.sim_loss(&c, s2, budget, d)
            };
            self.shadow_pos.unmake(&su, &mut self.shadow_hist);
            r
        } else {
            None
        };
        self.undo(mv, &st);
        let attrs = self.win_attrs(r?);
        self.store_proof(me, true, attrs, hint as u8, start - *budget);
        Some(attrs)
    }

    /// Simulation at a node where the original side is to move (`me`, the
    /// current position): every legal move must lose, as in `mid_inner`.
    fn sim_loss(&mut self, me: &Child, ssig: u64, budget: &mut u64, d: usize) -> Option<Attrs> {
        if *budget == 0 || self.hist.len() >= MAX_PLY {
            return None;
        }
        let start = *budget;
        *budget -= 1;
        while self.sim_bufs.len() <= d {
            self.sim_bufs.push(Vec::with_capacity(NPOINTS + 1));
        }
        let mut kids = std::mem::take(&mut self.sim_bufs[d]);
        let (win_now, excl, merged) = self.generate(&mut kids);
        let mut ok = win_now.is_none();
        if ok {
            for k in &mut kids {
                if k.phi == 0 {
                    continue;
                }
                let c = *k;
                let mv = Move::<ROWS, COLS>::decode(c.mv as usize).expect("valid move code");
                let st = self.do_move(mv).expect("child move was legal at generation");
                let r = if let Ok(su) = self.shadow_pos.play(mv, &mut self.shadow_hist) {
                    let s2 = if matches!(mv, Move::Stone(_)) { ssig ^ Self::sig_terms(self.shadow_pos.board().hash(), self.shadow_pos.board().hash2()).0 } else { ssig };
                    let r = self.sim_win(&c, s2, budget, d + 1);
                    self.shadow_pos.unmake(&su, &mut self.shadow_hist);
                    r
                } else {
                    None
                };
                self.undo(mv, &st);
                if let Some(a) = r {
                    k.phi = 0;
                    k.delta = INF;
                    k.attrs = a;
                } else {
                    ok = false;
                    break;
                }
            }
        }
        let res = if ok {
            let mut a = Attrs { target: excl, ..Attrs::empty::<NPOINTS>() };
            for c in &kids {
                a = a.join(c.attrs);
            }
            if merged {
                a.target = Target::Many(MANY_MERGE | a.target.many_causes());
            }
            let attrs = self.loss_attrs(a);
            self.store_proof(me, false, attrs, NPOINTS as u8, start - *budget);
            Some(attrs)
        } else {
            None
        };
        self.sim_bufs[d] = kids;
        res
    }

    /// The winning move stored for the shadow position, in its raw frame, from
    /// any proven entry of it: base, alternate or this line's twin. A hint only:
    /// the simulation verifies whatever it plays.
    fn shadow_hint(&self, ssig: u64) -> Option<usize> {
        let b = self.shadow_pos.board();
        let mut canon = u64::MAX;
        let mut sym = 0;
        for (s, &symmetry) in SYMS.iter().enumerate().take(self.nsym) {
            let h = b.hash_under_symmetry(symmetry);
            if h < canon {
                canon = h;
                sym = s;
            }
        }
        let stm = match self.shadow_pos.to_move() {
            Color::Black => STM_BLACK,
            Color::White => STM_WHITE,
        };
        let passes = if self.shadow_pos.consecutive_passes() > 0 { PASS_KEY } else { 0 };
        let key = canon ^ stm ^ passes;
        for k in [key, key ^ ALT_KEYS[0], key ^ ALT_KEYS[1], key ^ ssig] {
            if let Some(e) = self.tt.probe(k)
                && e.status == PnStatus::Win
            {
                let pv = e.pv as usize;
                return Some(if pv < NPOINTS { self.sym_map[sym_inverse(sym)][pv] } else { pv });
            }
        }
        None
    }

    /// A stored proof that the side to move wins the current position *on this
    /// path*, under the same test as `classify`: the base entry or an alternate
    /// if it transfers, else this path's twin in the same frame. (Another history's proof of the
    /// same canonical position may name a move that does not win here.)
    fn stored_win(&mut self) -> Option<PnEntry> {
        let (canon, sym) = self.canon();
        let key = self.key_of(canon);
        let k2 = self.key2(sym);
        for k in [key, key ^ ALT_KEYS[0], key ^ ALT_KEYS[1]] {
            if let Some(e) = self.tt.probe(k).filter(|e| e.status == PnStatus::Win && e.check == k2)
                && self.transfers(&e, sym)
            {
                return Some(e);
            }
        }
        self.tt
            .probe(key ^ self.sig)
            .filter(|e| e.status == PnStatus::Win && e.many && e.check == k2 ^ self.sig2 && e.sym == sym && e.sig == self.sig)
    }

    /// The winning move of `stored_win`, mapped back to the current frame.
    fn stored_win_move(&mut self) -> Option<usize> {
        let e = self.stored_win()?;
        let sym = self.canon().1;
        Some(if (e.pv as usize) < NPOINTS {
            self.sym_map[sym_inverse(sym as usize)][e.pv as usize]
        } else {
            e.pv as usize
        })
    }

    /// A principal variation of the stored proof that `winner` wins from the
    /// current position: the winner's stored moves and, at the loser's turns,
    /// the reply whose stored proof took the most work (the most resistant).
    /// Stops at a terminal position, where no stored move exists (static or
    /// evicted), or after `max` plies. Leaves the searcher at the start.
    fn principal_variation(&mut self, winner: Color, max: usize) -> Vec<usize> {
        let mut line = Vec::new();
        let mut stack = Vec::new();
        while line.len() < max && !self.pos.is_terminal() {
            let code = if self.pos.to_move() == winner {
                match self.stored_win_move() {
                    Some(c) => c,
                    None => break,
                }
            } else {
                let mut best: Option<(u64, usize)> = None;
                for k in 0..=NPOINTS {
                    let Some(mv) = Move::<ROWS, COLS>::decode(k) else { continue };
                    let Ok(st) = self.do_move(mv) else { continue };
                    let work = self.stored_win().map(|e| e.work);
                    self.undo(mv, &st);
                    if let Some(w) = work
                        && best.is_none_or(|(bw, _)| w > bw)
                    {
                        best = Some((w, k));
                    }
                }
                match best {
                    Some((_, k)) => k,
                    None => break,
                }
            };
            let Some(mv) = Move::<ROWS, COLS>::decode(code) else { break };
            let Ok(st) = self.do_move(mv) else { break };
            line.push(code);
            stack.push((mv, st));
        }
        while let Some((mv, st)) = stack.pop() {
            self.undo(mv, &st);
        }
        line
    }

    fn report(&mut self, depth: usize, total: u64) {
        self.next_report = total + if self.cfg.on_progress.is_some() { 1 << 18 } else { 1 << 23 };
        if !self.cfg.progress {
            return;
        }
        if let Some(f) = self.cfg.on_progress {
            // A snapshot samples the tables: build one at most 4 times a second.
            let elapsed = self.started.elapsed().as_secs_f64();
            if elapsed < self.next_progress {
                return;
            }
            self.next_progress = elapsed + 0.25;
            // Other workers' progress on open moves, from the shared table
            // (unproven entries only: a stored proof may not hold on this path).
            let root_moves = self
                .root_kids
                .iter()
                .map(|&(mv, p, d, key)| {
                    let (p, d) = if p == 0 || d == 0 {
                        (p, d)
                    } else {
                        self.tt
                            .probe(key)
                            .filter(|e| e.status == PnStatus::Unproven)
                            .map_or((p, d), |e| (e.phi, e.delta))
                    };
                    (usize::from(mv), p, d)
                })
                .collect();
            let root_symmetries =
                (1..self.nsym).filter(|&s| self.root_sym_mask & (1 << s) != 0).map(|s| self.sym_map[s].clone()).collect();
            f(&Progress {
                elapsed,
                nodes: total,
                root_phi: self.root_pd.0,
                root_delta: self.root_pd.1,
                max_ply: self.stats.max_ply,
                root_moves,
                root_symmetries,
                table_fill: self.tt.fill(4096),
                static_fill: self.sc.fill(4096),
            });
            return;
        }
        if (total >> 23).is_multiple_of(16) && !self.root_kids.is_empty() {
            let show = |v: u64| match v {
                0 => "0".to_string(),
                INF => "inf".to_string(),
                x => x.to_string(),
            };
            let parts: Vec<String> = self
                .root_kids
                .iter()
                .map(|&(mv, p, d, key)| {
                    let m = Move::<ROWS, COLS>::decode(mv as usize).map_or_else(|| "?".to_string(), |m| match m {
                        Move::Pass => "pass".to_string(),
                        Move::Stone(pt) => format!("{}{}", pt.row(), pt.col()),
                    });
                    // Other workers' progress on open moves, from the shared table
                    // (unproven entries only: a stored proof may not hold on this path).
                    let (p, d) = if p == 0 || d == 0 {
                        (p, d)
                    } else {
                        self.tt
                            .probe(key)
                            .filter(|e| e.status == PnStatus::Unproven)
                            .map_or((p, d), |e| (e.phi, e.delta))
                    };
                    format!("{m}:({},{})", show(p), show(d))
                })
                .collect();
            eprintln!("[root] {}", parts.join(" "));
            let path: Vec<String> = self
                .path_pd
                .iter()
                .map(|&(mv, p, d)| {
                    let m = Move::<ROWS, COLS>::decode(mv as usize).map_or_else(|| "?".to_string(), |m| match m {
                        Move::Pass => "pass".to_string(),
                        Move::Stone(pt) => format!("{}{}", pt.row(), pt.col()),
                    });
                    format!("({},{})->{m}", show(p), show(d))
                })
                .collect();
            eprintln!("[path] {}", path.join(" "));
        }
        let s = &self.stats;
        let el = self.started.elapsed().as_secs_f64();
        eprintln!(
            "[dfpn] {:>8.1}s nodes={} ({:.0}/s) root={:?} kids(t0)={} depth={} maxply={} tt_hit={:.1}% reused={} rej tgt/bloom={}/{} twin hit/store={}/{} psk={}",
            el,
            total,
            total as f64 / el.max(1e-9),
            self.root_pd,
            s.children,
            depth,
            s.max_ply,
            100.0 * s.tt_hits as f64 / (s.tt_probes.max(1)) as f64,
            s.reuse_checked,
            s.reject_target,
            s.reject_bloom,
            s.twin_hits,
            s.twin_stores,
            s.psk_exclusions,
        );
    }
}

/// Solve the position reached by `prefix` from the empty board (whose boards
/// form the superko history) by df-pn. Runs on a dedicated thread
/// with a large stack (the recursion depth follows the game length).
#[must_use]
pub fn solve_dfpn<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
) -> DfpnResult {
    let prefix = prefix.to_vec();
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || solve_inner::<ROWS, COLS, LANES, NPOINTS>(&prefix, cfg))
        .expect("spawn solver thread")
        .join()
        .expect("solver thread panicked")
}

fn solve_inner<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
) -> DfpnResult {
    // At most 49 points: the 128-bit board key and the 128-bit position key
    // (base key and second hash) are then exact, because their generators are
    // linearly independent (tested in `sanko-core`'s `zobrist.rs` and below).
    // So the superko check, the stabilizer, the path signatures' per-board
    // inputs and the identity of proof entries are exact. Up to 30 points the
    // 64-bit halves alone are exact, and the second board table is not kept.
    assert!(NPOINTS <= 49, "boards above 49 points need wider keys");
    let layout = crate::pntt::Layout::for_points(NPOINTS);
    let tt = if cfg.proof_buckets > 0 {
        PnTable::with_sizes(cfg.tt_log_buckets, cfg.proof_buckets as usize, layout, cfg.threads)
    } else {
        PnTable::with_layout(cfg.tt_log_buckets, cfg.proof_log, layout, cfg.threads)
    };
    let sc = StaticCache::new(cfg.static_cache_log_slots, cfg.threads);
    let sh = Shared::new(cfg.threads);
    let mut res = search_all::<ROWS, COLS, LANES, NPOINTS>(prefix, &tt, &sc, cfg, &sh);
    {
        let quiet = Shared::new(1);
        let mut s = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, &tt, &sc, cfg, &quiet, 1);
        if res.winner == Some(s.pos.to_move()) {
            res.root_move = s.stored_win_move();
        }
        if let Some(w) = res.winner {
            res.pv = s.principal_variation(w, 300);
        }
    }
    res
}

fn search_all<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    tt: &PnTable,
    sc: &StaticCache,
    cfg: DfpnConfig,
    sh: &Shared,
) -> DfpnResult {
    if cfg.threads <= 1 {
        return run_worker::<ROWS, COLS, LANES, NPOINTS>(prefix, tt, sc, cfg, sh, 0);
    }
    let root_stm = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, tt, sc, cfg, sh, usize::MAX).pos.to_move();
    let results: Vec<DfpnResult> = std::thread::scope(|scope| {
        // Collect first: spawning lazily inside the join loop would serialize workers.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..cfg.threads)
            .map(|tid| {
                std::thread::Builder::new()
                    .stack_size(1 << 30)
                    .spawn_scoped(scope, move || run_worker::<ROWS, COLS, LANES, NPOINTS>(prefix, tt, sc, cfg, sh, tid))
                    .expect("spawn worker")
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let mut stats = DfpnStats::default();
    for r in &results {
        stats.merge(&r.stats);
    }
    let verdict = *sh.result.lock().expect("result lock");
    let winner = verdict.and_then(|stm_wins| reported(cfg, if stm_wins { root_stm } else { root_stm.opponent() }));
    DfpnResult { winner, stats, root_move: None, pv: Vec::new() }
}

/// The reported winner for a root verdict `w`: pruning side `c`'s moves makes
/// every disproof of `c` unsound, so a verdict against the pruned side is
/// reported as unknown (in every mode, including a root that is already
/// terminal or statically decided, to keep one rule).
fn reported(cfg: DfpnConfig, w: Color) -> Option<Color> {
    match cfg.prune_for {
        Some(c) if c != w => None,
        _ => Some(w),
    }
}

fn run_worker<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    tt: &PnTable,
    sc: &StaticCache,
    cfg: DfpnConfig,
    sh: &Shared,
    tid: usize,
) -> DfpnResult {
    let mut s = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, tt, sc, cfg, sh, tid);
    let root_stm = s.pos.to_move();

    // The root itself may already be decided.
    let (canon, _) = s.canon();
    let key = s.key_of(canon);
    // (The shared verdict is what `solve_dfpn` reports with several threads.)
    let decided = |w: Color| {
        let mut r = sh.result.lock().expect("result lock");
        if r.is_none() {
            *r = Some(w == root_stm);
        }
    };
    if s.pos.is_terminal() {
        let w = terminal_winner(&s.pos, cfg.komi_times_two).unwrap_or(Color::White);
        decided(w);
        return DfpnResult { winner: reported(cfg, w), stats: s.stats, root_move: None, pv: Vec::new() };
    }
    if let Some(w) = sc
        .verdict::<ROWS, COLS, LANES, NPOINTS>(key, &s.pos, cfg.komi_times_two)
        .filter(|&w| !cfg.no_static && s.static_leaf_terr(w).is_some())
    {
        decided(w);
        return DfpnResult { winner: reported(cfg, w), stats: s.stats, root_move: None, pv: Vec::new() };
    }
    let root = Child {
        mv: 0,
        phi: 1,
        delta: 1,
        key,
        base_key: key,
        twin: false,
        attrs: Attrs::UNPROVEN,
        fresh: false,
    };
    let mut last = (u64::MAX, u64::MAX);
    let mut stuck = 0;
    loop {
        let (phi, delta, _) = s.mid(root, INF - 1, INF - 1, 0);
        if phi == 0 || delta == 0 {
            {
                let mut r = sh.result.lock().expect("result lock");
                if r.is_none() {
                    *r = Some(phi == 0);
                }
            }
            sh.stop.store(true, Ordering::Relaxed);
            let w = if phi == 0 { root_stm } else { root_stm.opponent() };
            return DfpnResult { winner: reported(cfg, w), stats: s.stats, root_move: None, pv: Vec::new() };
        }
        if s.aborted {
            return DfpnResult { winner: None, stats: s.stats, root_move: None, pv: Vec::new() };
        }
        if (phi, delta) == last {
            stuck += 1;
            if stuck > 1000 {
                eprintln!("[dfpn] no progress at root (phi={phi}, delta={delta}); giving up");
                return DfpnResult { winner: None, stats: s.stats, root_move: None, pv: Vec::new() };
            }
        } else {
            stuck = 0;
            last = (phi, delta);
        }
    }
}

/// Outcome of [`solve_dfpn_resumable`].
#[derive(Clone, Debug)]
pub struct ResumableResult {
    /// This segment's result. Its statistics cover this segment only.
    pub result: DfpnResult,
    /// The search stopped at `stop_after` before the root was decided.
    pub stopped: bool,
    /// The segment's number: 1 for a fresh run, else one more than the
    /// resumed checkpoint's.
    pub segment: u64,
    /// Expansions and seconds of the earlier segments (0 for a fresh run).
    pub earlier_nodes: u64,
    pub earlier_seconds: f64,
    /// Snapshots written by this segment.
    pub snapshots: u64,
    /// The error of the last snapshot, if it failed.
    pub snapshot_error: Option<String>,
    /// The checkpoint of an earlier position of the game that seeded this run.
    pub seeded_from: Option<std::path::PathBuf>,
    /// The root's children with their final numbers (see `root_children`),
    /// empty at a terminal root.
    pub root_moves: Vec<(usize, u64, u64)>,
    /// Seeded at another komi: the proofs kept and cleared (see
    /// [`crate::checkpoint`]).
    pub komi_seed: Option<(u64, u64)>,
}

/// The root's children as the search generates them: `(move code, phi,
/// delta)`, seen from the position after the move (delta 0 marks a winning
/// move, phi 0 a refuted one). Symmetric moves are merged under the symmetries
/// that fix every board of the game so far, and the pruned side's skipped moves
/// are left out, exactly as in the search. With a fresh table the numbers come
/// from static knowledge and the prior.
#[must_use]
pub fn root_children<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
) -> Vec<(usize, u64, u64)> {
    let layout = crate::pntt::Layout::for_points(NPOINTS);
    let tt = PnTable::with_layout(cfg.tt_log_buckets, cfg.proof_log, layout, 1);
    let sc = StaticCache::new(cfg.static_cache_log_slots, 1);
    let sh = Shared::new(1);
    let mut s = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, &tt, &sc, cfg, &sh, 1);
    if s.pos.is_terminal() { Vec::new() } else { s.children_numbers() }
}

/// Marks a worker as returned when dropped, also when it unwinds, so that a
/// snapshot never waits for a worker that is gone.
struct Leaving<'a>(&'a Shared);

impl Drop for Leaving<'_> {
    fn drop(&mut self) {
        self.0.leave();
    }
}

/// The checkpoint side of a resumable solve: periodic snapshots and the stop.
struct Snapshots<'a> {
    opts: &'a CheckpointOptions,
    header: CheckpointHeader,
    started: Instant,
    segment: u64,
    earlier_nodes: u64,
    earlier_seconds: f64,
    io_threads: usize,
    written: u64,
    error: Option<String>,
    stopped: bool,
}

impl Snapshots<'_> {
    /// Write a snapshot of `tt` while the workers are parked, if the run
    /// writes any.
    fn snapshot(&mut self, tt: &PnTable, sh: &Shared) {
        let Some(path) = &self.opts.write else { return };
        let mut h = self.header.clone();
        h.set("segments", self.segment);
        h.set("nodes", self.earlier_nodes + sh.nodes.load(Ordering::Relaxed));
        h.set("seconds", format!("{:.1}", self.earlier_seconds + self.started.elapsed().as_secs_f64()));
        h.set("written_unix", SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()));
        let t = Instant::now();
        match checkpoint::write_snapshot(tt, &h, path, self.io_threads) {
            Ok(()) => {
                self.written += 1;
                self.error = None;
                eprintln!(
                    "[checkpoint] wrote {} ({:.1} GiB in {:.1} s)",
                    path.display(),
                    tt.bytes() as f64 / f64::from(1u32 << 30),
                    t.elapsed().as_secs_f64()
                );
            }
            Err(e) => {
                eprintln!("[checkpoint] snapshot failed: {e}");
                self.error = Some(e.to_string());
            }
        }
    }

    /// Run until every worker has returned: a snapshot every `every` (with
    /// `write`), and at `stop_after` a last snapshot and the stop.
    fn coordinate(&mut self, tt: &PnTable, sh: &Shared) {
        let every = self.opts.write.as_ref().and(self.opts.every);
        let mut next = every.map(|d| Instant::now() + d);
        let mut stop_at = self.opts.stop_after.map(|d| self.started + d);
        loop {
            let now = Instant::now();
            let step = [next, stop_at]
                .into_iter()
                .flatten()
                .map(|t| t.saturating_duration_since(now))
                .fold(Duration::from_millis(500), Duration::min)
                .max(Duration::from_millis(1));
            if sh.wait_done(step) {
                if self.opts.at_end && !self.stopped {
                    // Every worker has returned, so the table no longer changes.
                    self.snapshot(tt, sh);
                }
                return;
            }
            let now = Instant::now();
            if stop_at.is_some_and(|t| now >= t) {
                if sh.pause_all() && !sh.stop.load(Ordering::Relaxed) {
                    self.snapshot(tt, sh);
                    self.stopped = true;
                }
                sh.stop.store(true, Ordering::Relaxed);
                sh.resume_all();
                (stop_at, next) = (None, None);
            } else if next.is_some_and(|t| now >= t) {
                if sh.pause_all() && !sh.stop.load(Ordering::Relaxed) {
                    self.snapshot(tt, sh);
                }
                sh.resume_all();
                next = every.map(|d| Instant::now() + d);
            }
        }
    }
}

/// `search_all` with a checkpoint coordinator on the calling thread. Workers
/// always run on their own threads, also a single one.
fn search_checkpointed<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    tt: &PnTable,
    sc: &StaticCache,
    cfg: DfpnConfig,
    sh: &Shared,
    snaps: &mut Snapshots<'_>,
) -> DfpnResult {
    let root_stm = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, tt, sc, cfg, sh, usize::MAX).pos.to_move();
    let workers = cfg.threads.max(1);
    sh.gate.lock().expect("gate lock").active = workers;
    let results: Vec<DfpnResult> = std::thread::scope(|scope| {
        // Collect first: spawning lazily inside the join loop would serialize workers.
        #[allow(clippy::needless_collect)]
        let handles: Vec<_> = (0..workers)
            .map(|tid| {
                std::thread::Builder::new()
                    .stack_size(1 << 30)
                    .spawn_scoped(scope, move || {
                        let _leaving = Leaving(sh);
                        run_worker::<ROWS, COLS, LANES, NPOINTS>(prefix, tt, sc, cfg, sh, tid)
                    })
                    .expect("spawn worker")
            })
            .collect();
        snaps.coordinate(tt, sh);
        handles.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let mut stats = DfpnStats::default();
    for r in &results {
        stats.merge(&r.stats);
    }
    let verdict = *sh.result.lock().expect("result lock");
    let winner = verdict.and_then(|stm_wins| reported(cfg, if stm_wins { root_stm } else { root_stm.opponent() }));
    DfpnResult { winner, stats, root_move: None, pv: Vec::new() }
}

/// The settings that a checkpoint of this run records. Those of
/// `checkpoint::MUST_MATCH` must agree for a resume. The rest steer the search
/// without changing what a proof means and are recorded for reference.
fn run_header<const ROWS: usize, const COLS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
    build: &str,
    layout: crate::pntt::Layout,
    proof_buckets: usize,
) -> CheckpointHeader {
    let mut h = CheckpointHeader::default();
    h.set("build", build);
    h.set("board", format!("{ROWS}x{COLS}"));
    h.set("komi_times_two", cfg.komi_times_two);
    h.set("moves", prefix.iter().map(|m| m.encode().to_string()).collect::<Vec<_>>().join(","));
    h.set(
        "prune_for",
        match cfg.prune_for {
            Some(Color::Black) => "black",
            Some(Color::White) => "white",
            None => "none",
        },
    );
    h.set("keep_tempo", cfg.keep_tempo);
    h.set("exact_only", cfg.exact_only);
    h.set("no_static", cfg.no_static);
    h.set("region_solver", cfg.region_solver);
    h.set("region_max", cfg.region_max);
    h.set("region_interior", cfg.region_interior);
    h.set("atari_dead", sanko_core::atari_dead());
    h.set("layout", layout.name());
    h.set("unproven_buckets", 1u64 << cfg.tt_log_buckets);
    h.set("proof_buckets", proof_buckets);
    h.set("eps", cfg.eps);
    h.set("simulate", cfg.simulate);
    h.set("threads", cfg.threads);
    h
}

/// [`solve_dfpn`] with checkpoints (see [`crate::checkpoint`]). It continues
/// from `opts.resume` when given (the same build, problem and table sizes, or
/// an error), writes a snapshot to `opts.write` every `opts.every`, and after
/// `opts.stop_after` writes a last one and stops. A snapshot pauses the search
/// without changing it, so serial runs stay deterministic between resumes.
pub fn solve_dfpn_resumable<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
    opts: &CheckpointOptions,
) -> Result<ResumableResult, CheckpointError> {
    let prefix = prefix.to_vec();
    let opts = opts.clone();
    std::thread::Builder::new()
        .stack_size(1 << 30)
        .spawn(move || resumable_inner::<ROWS, COLS, LANES, NPOINTS>(&prefix, cfg, &opts))
        .expect("spawn solver thread")
        .join()
        .expect("solver thread panicked")
}

fn resumable_inner<const ROWS: usize, const COLS: usize, const LANES: usize, const NPOINTS: usize>(
    prefix: &[Move<ROWS, COLS>],
    cfg: DfpnConfig,
    opts: &CheckpointOptions,
) -> Result<ResumableResult, CheckpointError> {
    assert!(NPOINTS <= 49, "boards above 49 points need wider keys");
    let started = Instant::now();
    let layout = crate::pntt::Layout::for_points(NPOINTS);
    let proof_buckets = if cfg.proof_buckets > 0 { cfg.proof_buckets as usize } else { 1usize << cfg.proof_log };
    let header = run_header(prefix, cfg, &opts.build, layout, proof_buckets);
    if let Some(w) = &opts.write
        && w.exists()
        && !opts.resume.as_deref().is_some_and(|r| checkpoint::same_file(r, w))
    {
        return Err(CheckpointError::Refused(format!(
            "{} exists and is not the checkpoint being resumed: resume from it or remove it",
            w.display()
        )));
    }
    let mut header = header;
    let resumed = match &opts.resume {
        Some(path) => {
            let h = checkpoint::read_checkpoint_header(path)?;
            let mut diff = h.mismatches(&header);
            let moves = |h: &CheckpointHeader| -> Vec<String> {
                h.get("moves").unwrap_or("").split(',').filter(|m| !m.is_empty()).map(str::to_string).collect()
            };
            let (theirs, ours) = (moves(&h), moves(&header));
            let prefix_seed = theirs.len() < ours.len() && ours.starts_with(&theirs);
            // Another komi keeps the proofs that stay valid at it: Black's wins
            // below the checkpoint's komi, White's above, each only if the
            // checkpoint's run was not pruned for the other side.
            let komi = |h: &CheckpointHeader| h.get("komi_times_two").and_then(|v| v.parse::<i32>().ok());
            let (k_old, k_new) = (komi(&h), komi(&header));
            let (p_old, p_new) = (h.get("prune_for").unwrap_or("-"), header.get("prune_for").unwrap_or("-"));
            let mut komi_seed = None;
            match (k_old, k_new) {
                (Some(a), Some(b)) if a != b => komi_seed = Some((b < a && p_old != "white", b > a && p_old != "black")),
                (Some(_), Some(_)) if p_old != p_new => {
                    diff.push(format!("prune_for: {p_old} in the checkpoint, {p_new} in this run (only another komi may change it)"));
                }
                (Some(_), Some(_)) => {}
                _ => diff.push("komi_times_two: missing".to_string()),
            }
            let seeded = prefix_seed || komi_seed.is_some();
            if theirs != ours && !prefix_seed {
                diff.push(format!(
                    "moves: {} in the checkpoint, {} in this run (it must be this position or an earlier one of the same game)",
                    h.get("moves").unwrap_or(""),
                    header.get("moves").unwrap_or("")
                ));
            }
            if seeded {
                header.set(
                    "seed",
                    format!(
                        "{} at moves {} komi_times_two {}",
                        path.display(),
                        h.get("moves").unwrap_or(""),
                        h.get("komi_times_two").unwrap_or("")
                    ),
                );
            } else if let Some(seed) = h.get("seed") {
                header.set("seed", seed);
            }
            if !diff.is_empty() {
                return Err(CheckpointError::Refused(format!(
                    "{} belongs to a different run:\n  {}",
                    path.display(),
                    diff.join("\n  ")
                )));
            }
            Some((path, h, seeded, komi_seed))
        }
        None => None,
    };
    let io_threads = cfg.threads.clamp(1, 32);
    let tt = PnTable::with_sizes(cfg.tt_log_buckets, proof_buckets, layout, cfg.threads);
    let seeded_from = resumed.as_ref().filter(|r| r.2).map(|r| r.0.clone());
    let mut komi_seed = None;
    let (segment, earlier_nodes, earlier_seconds) = match &resumed {
        Some((path, h, seeded, keep)) => {
            let t = Instant::now();
            checkpoint::load_snapshot(&tt, h, path, io_threads)?;
            if let Some((keep_black, keep_white)) = *keep {
                let (kept, cleared) =
                    tt.retain_proofs(|black_wins| if black_wins { keep_black } else { keep_white }, io_threads);
                let which = match (keep_black, keep_white) {
                    (true, _) => "Black's wins",
                    (_, true) => "White's wins",
                    _ => "no proofs",
                };
                eprintln!("[checkpoint] another komi: kept {kept} proofs ({which}), cleared {cleared}");
                komi_seed = Some((kept, cleared));
            }
            eprintln!(
                "[checkpoint] resumed {} ({:.1} GiB in {:.1} s)",
                path.display(),
                tt.bytes() as f64 / f64::from(1u32 << 30),
                t.elapsed().as_secs_f64()
            );
            if *seeded {
                (1, 0, 0.0)
            } else {
                (h.number("segments") + 1, h.number("nodes"), h.get("seconds").and_then(|v| v.parse().ok()).unwrap_or(0.0))
            }
        }
        None => (1, 0, 0.0),
    };
    let sc = StaticCache::new(cfg.static_cache_log_slots, cfg.threads);
    let sh = Shared::new(cfg.threads);
    let mut snaps = Snapshots {
        opts,
        header,
        started,
        segment,
        earlier_nodes,
        earlier_seconds,
        io_threads,
        written: 0,
        error: None,
        stopped: false,
    };
    let mut res = search_checkpointed::<ROWS, COLS, LANES, NPOINTS>(prefix, &tt, &sc, cfg, &sh, &mut snaps);
    let quiet = Shared::new(1);
    let mut s = Searcher::<ROWS, COLS, LANES, NPOINTS>::new(prefix, &tt, &sc, cfg, &quiet, 1);
    if let Some(w) = res.winner {
        if w == s.pos.to_move() {
            res.root_move = s.stored_win_move();
        }
        res.pv = s.principal_variation(w, 300);
    }
    let root_moves = if s.pos.is_terminal() { Vec::new() } else { s.children_numbers() };
    Ok(ResumableResult {
        result: res,
        stopped: snaps.stopped,
        segment,
        earlier_nodes,
        earlier_seconds,
        snapshots: snaps.written,
        snapshot_error: snaps.error,
        seeded_from,
        root_moves,
        komi_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 1×5 mirror pair of `tests/dfpn.rs` (komi -2.5) on ONE table: after
    /// one game is solved, every prefix of the other game is solved on the same
    /// table and must agree with a fresh exact-only solve. The second search
    /// meets the first game's path-specific proofs under the same twin keys
    /// (mirror-image boards, the same board set) and must reject them by frame.
    /// Separate solves cannot exercise this.
    #[test]
    fn mirror_endpoints_on_a_shared_table() {
        let line = |s: &str| -> Vec<Move<1, 5>> {
            s.split_whitespace()
                .map(|t| if t == "P" { Move::Pass } else { Move::Stone(Point::<1, 5>::new(0, t.parse().unwrap()).unwrap()) })
                .collect()
        };
        let a = line("1 P 0 2 0 1 P 0 3 2 P 4 P 3");
        let b = line("3 2 P 4 P 3 1 P 0 2 0 1 P 0");
        let cfg = DfpnConfig {
            komi_times_two: -5,
            tt_log_buckets: 16,
            proof_log: 15,
            static_cache_log_slots: 16,
            ..DfpnConfig::default()
        };
        let oracle = DfpnConfig { exact_only: true, no_static: true, simulate: 0, ..cfg };
        let mut frame_rejects = 0;
        for (first, second) in [(&a, &b), (&b, &a)] {
            for k in 1..=second.len() {
                let tt = PnTable::new(cfg.tt_log_buckets, cfg.proof_log, false);
                let sc = StaticCache::new(cfg.static_cache_log_slots, 1);
                search_all::<1, 5, 1, 5>(first, &tt, &sc, cfg, &Shared::new(1));
                let r = search_all::<1, 5, 1, 5>(&second[..k], &tt, &sc, cfg, &Shared::new(1));
                let want = solve_dfpn::<1, 5, 1, 5>(&second[..k], oracle).winner;
                assert_eq!(r.winner, want, "shared table, prefix of length {k}");
                frame_rejects += r.stats.twin_frame_rejects;
            }
        }
        assert!(frame_rejects > 0, "the second game should meet mirrored twin proofs");
    }

    /// A random game of up to `plies` moves (single passes allowed, never two in a row).
    fn random_line<const R: usize, const C: usize, const L: usize, const N: usize>(
        rng: &mut sanko_core::Rng,
        plies: u32,
    ) -> Vec<Move<R, C>> {
        let mut pos: Position<R, C, L, N> = Position::new();
        let mut hist = HashHistory::new();
        hist.push(pos.board().key());
        let mut moves = Vec::new();
        for _ in 0..plies {
            let mut legal: Vec<Move<R, C>> = (0..N)
                .map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k)))
                .filter(|&m| pos.legal(m, &hist))
                .collect();
            if pos.consecutive_passes() == 0 && (legal.is_empty() || rng.next_below(4) == 0) {
                legal.push(Move::Pass);
            }
            if legal.is_empty() {
                break;
            }
            let mv = legal[rng.next_below(legal.len() as u32) as usize];
            pos.play(mv, &mut hist).unwrap();
            moves.push(mv);
        }
        moves
    }

    /// Every prefix of random games, solved one after another on ONE table (so
    /// proofs stored under one history meet the others, mirror images included),
    /// agrees with a fresh exact-only solve without static knowledge.
    fn shared_table_check<const R: usize, const C: usize, const L: usize, const N: usize>(
        seed: u64,
        games: usize,
        plies: u32,
        komis: &[i32],
    ) -> (u64, u64) {
        let mut rng = sanko_core::Rng::new(seed);
        let (mut compared, mut rejects) = (0, 0);
        for &k2 in komis {
            let cfg = DfpnConfig {
                komi_times_two: k2,
                tt_log_buckets: 16,
                proof_log: 15,
                static_cache_log_slots: 16,
                region_solver: true,
                node_limit: 2_000_000,
                ..DfpnConfig::default()
            };
            let oracle = DfpnConfig { exact_only: true, no_static: true, region_solver: false, simulate: 0, ..cfg };
            let tt = PnTable::new(cfg.tt_log_buckets, cfg.proof_log, false);
            let sc = StaticCache::new(cfg.static_cache_log_slots, 1);
            for _ in 0..games {
                let line = random_line::<R, C, L, N>(&mut rng, plies);
                for n in 0..=line.len() {
                    let r = search_all::<R, C, L, N>(&line[..n], &tt, &sc, cfg, &Shared::new(1));
                    rejects += r.stats.twin_frame_rejects;
                    let want = solve_dfpn::<R, C, L, N>(&line[..n], oracle).winner;
                    if let (Some(x), Some(y)) = (r.winner, want) {
                        assert_eq!(x, y, "{R}x{C} komi2={k2} prefix {n} of {line:?}");
                        compared += 1;
                    }
                }
            }
        }
        (compared, rejects)
    }

    /// Minimax under positional superko with an exact memo. A game state is the raw
    /// board, the side to move, the pass count and the set of earlier boards (board
    /// hashes are exact on these boards), so the memo needs no canonical frame, no
    /// twin keys and no transfer test. It shares only the rules engine and the
    /// scoring with the solver, so it cannot share a bug in the table logic (unlike
    /// the exact-only oracle, which reuses path-specific proofs through the same probe).
    struct BruteForce {
        memo: std::collections::HashMap<(u64, Color, u8, Vec<u64>), bool>,
        /// Boards of the game so far (the current one last).
        boards: Vec<u64>,
        budget: u64,
    }

    impl BruteForce {
        /// Does the side to move win? `None` once the budget of expansions is used up.
        fn wins<const R: usize, const C: usize, const L: usize, const N: usize>(
            &mut self,
            pos: &mut Position<R, C, L, N>,
            hist: &mut HashHistory,
            komi_times_two: i32,
        ) -> Option<bool> {
            if pos.is_terminal() {
                let w = terminal_winner(pos, komi_times_two).unwrap_or(Color::White);
                return Some(w == pos.to_move());
            }
            let mut set = self.boards.clone();
            set.sort_unstable();
            let key = (pos.board().hash(), pos.to_move(), pos.consecutive_passes(), set);
            if let Some(&win) = self.memo.get(&key) {
                return Some(win);
            }
            if self.budget == 0 {
                return None;
            }
            self.budget -= 1;
            let mut unknown = false;
            let mut win = false;
            let moves = (0..N).map(|k| Move::Stone(Point::<R, C>::from_index_unchecked(k))).chain([Move::Pass]);
            for mv in moves {
                let Ok(undo) = pos.play(mv, hist) else { continue };
                let stone = matches!(mv, Move::Stone(_));
                if stone {
                    self.boards.push(pos.board().hash());
                }
                let child = self.wins(pos, hist, komi_times_two);
                if stone {
                    self.boards.pop();
                }
                pos.unmake(&undo, hist);
                match child {
                    Some(false) => {
                        win = true;
                        break;
                    }
                    Some(true) => {}
                    None => unknown = true,
                }
            }
            if !win && unknown {
                return None;
            }
            self.memo.insert(key, win);
            Some(win)
        }
    }

    /// The winner after `line` by brute force, or `None` if it needs more than `budget` expansions.
    fn brute_force_winner<const R: usize, const C: usize, const L: usize, const N: usize>(
        line: &[Move<R, C>],
        komi_times_two: i32,
        budget: u64,
    ) -> Option<Color> {
        let mut pos: Position<R, C, L, N> = Position::new();
        let mut hist = HashHistory::new();
        hist.push(pos.board().key());
        let mut bf = BruteForce { memo: std::collections::HashMap::new(), boards: vec![pos.board().hash()], budget };
        for &mv in line {
            pos.play(mv, &mut hist).unwrap();
            if matches!(mv, Move::Stone(_)) {
                bf.boards.push(pos.board().hash());
            }
        }
        let stm = pos.to_move();
        bf.wins(&mut pos, &mut hist, komi_times_two).map(|win| if win { stm } else { stm.opponent() })
    }

    /// Every prefix of random games, solved one after another on ONE table (full
    /// knowledge, so proofs of one history meet the others, mirror images
    /// included), agrees with brute-force minimax wherever that finishes. The
    /// brute force runs from the longest prefix down and stops at the first one over
    /// budget, since shorter prefixes have larger game trees.
    #[allow(clippy::too_many_arguments)]
    fn brute_force_check<const R: usize, const C: usize, const L: usize, const N: usize>(
        seed: u64,
        games: usize,
        plies: u32,
        komis: &[i32],
        budget: u64,
        prior: Option<[f32; PRIOR_FEATURES]>,
        scale: f32,
        mid: bool,
    ) -> u64 {
        let sim = DfpnConfig::default().simulate;
        brute_force_check_with::<R, C, L, N>(seed, games, plies, komis, budget, prior, scale, mid, sim).0
    }

    /// `brute_force_check` with simulation budget `simulate`; also returns the
    /// number of successful simulations.
    #[allow(clippy::too_many_arguments)]
    fn brute_force_check_with<const R: usize, const C: usize, const L: usize, const N: usize>(
        seed: u64,
        games: usize,
        plies: u32,
        komis: &[i32],
        budget: u64,
        prior: Option<[f32; PRIOR_FEATURES]>,
        scale: f32,
        mid: bool,
        simulate: u64,
    ) -> (u64, u64) {
        let mut rng = sanko_core::Rng::new(seed);
        let (mut compared, mut sims) = (0, 0);
        for &k2 in komis {
            let cfg = DfpnConfig {
                komi_times_two: k2,
                tt_log_buckets: 16,
                proof_log: 15,
                static_cache_log_slots: 16,
                region_solver: true,
                node_limit: 2_000_000,
                prior_weights: prior,
                prior_scale: scale,
                simulate,
                ..DfpnConfig::default()
            };
            // The mid layout (any board up to 32 points) in a table whose size
            // is not a power of two, or the narrow layout of these boards.
            let tt = if mid {
                PnTable::with_sizes(cfg.tt_log_buckets, 3 << (cfg.proof_log - 2), crate::pntt::Layout::Mid, 1)
            } else {
                PnTable::new(cfg.tt_log_buckets, cfg.proof_log, false)
            };
            let sc = StaticCache::new(cfg.static_cache_log_slots, 1);
            for _ in 0..games {
                let line = random_line::<R, C, L, N>(&mut rng, plies);
                let mut truth = vec![None; line.len() + 1];
                for n in (0..=line.len()).rev() {
                    truth[n] = brute_force_winner::<R, C, L, N>(&line[..n], k2, budget);
                    if truth[n].is_none() {
                        break;
                    }
                }
                for (n, want) in truth.iter().enumerate() {
                    let r = search_all::<R, C, L, N>(&line[..n], &tt, &sc, cfg, &Shared::new(1));
                    sims += r.stats.sim_successes;
                    if let (Some(x), Some(y)) = (r.winner, *want) {
                        assert_eq!(x, y, "{R}x{C} komi2={k2} prefix {n} of {line:?}");
                        compared += 1;
                    }
                }
            }
        }
        (compared, sims)
    }

    #[test]
    fn simulation_keeps_brute_force_results() {
        // A budget of one node (almost every attempt fails half-way) and an
        // unlimited one (attempts run to the end of the replayed line), and the
        // simulation must actually prove nodes.
        let mut compared = 0;
        for sim in [1, 1_000_000] {
            let (a, s1) = brute_force_check_with::<1, 5, 1, 5>(31, 20, 24, &[-5, -3, -1, 1], 200_000, None, 0.0, false, sim);
            let (b, s2) = brute_force_check_with::<2, 3, 1, 6>(33, 12, 16, &[-1, 1, 3], 200_000, None, 0.0, true, sim);
            let (c, s3) = brute_force_check_with::<3, 3, 1, 9>(35, 6, 14, &[5, 13, 17], 200_000, None, 0.0, false, sim);
            assert!(sim == 1 || s1 + s2 + s3 > 0, "no successful simulation");
            compared += a + b + c;
        }
        assert!(compared > 400, "only {compared} positions compared");
        eprintln!("simulation brute-force check: {compared} positions compared");
    }

    #[test]
    fn shared_table_agrees_with_brute_force_minimax() {
        let compared = brute_force_check::<1, 4, 1, 4>(11, 30, 20, &[-3, -1, 1, 3, 7], 200_000, None, 0.0, false)
            + brute_force_check::<1, 5, 1, 5>(12, 20, 24, &[-5, -3, -1, 1], 200_000, None, 0.0, false)
            + brute_force_check::<2, 3, 1, 6>(13, 12, 16, &[-1, 1, 3], 200_000, None, 0.0, false);
        assert!(compared > 500, "only {compared} positions compared");
        eprintln!("brute-force check: {compared} positions compared");
    }

    #[test]
    fn learned_prior_keeps_brute_force_results() {
        // The prior only orders the search: with the built-in model, an
        // arbitrary one and an inverted one the result must be the minimax value.
        let w = [0.5, 1.0, -3.0, 4.0, -4.0, -2.0, -1.0, 1.0, -0.5, 0.5, -0.2, 0.2, 1.0, -1.0];
        let inv = w.map(|x: f32| -x);
        let mut compared = 0;
        for e in [crate::prior::DEFAULT_PRIOR_WEIGHTS, w, inv] {
            compared += brute_force_check::<1, 4, 1, 4>(21, 20, 20, &[-3, -1, 1, 3], 200_000, Some(e), 6.0, false)
                + brute_force_check::<2, 3, 1, 6>(23, 10, 16, &[-1, 1, 3], 200_000, Some(e), 6.0, false);
        }
        assert!(compared > 300, "only {compared} positions compared");
        eprintln!("learned-prior brute-force check: {compared} positions compared");
    }

    #[test]
    fn mid_layout_and_odd_table_size_keep_brute_force_results() {
        // The search on the mid layout in a table of 3·2^13 buckets, with the
        // built-in prior at the default scale, and with no prior.
        let w = Some(crate::prior::DEFAULT_PRIOR_WEIGHTS);
        let s = crate::prior::DEFAULT_PRIOR_SCALE;
        let compared = brute_force_check::<1, 4, 1, 4>(41, 20, 20, &[-3, -1, 1, 3], 200_000, w, s, true)
            + brute_force_check::<2, 3, 1, 6>(43, 10, 16, &[-1, 1, 3], 200_000, w, s, true)
            + brute_force_check::<1, 5, 1, 5>(42, 12, 24, &[-5, -1, 1], 200_000, None, 0.0, true);
        assert!(compared > 300, "only {compared} positions compared");
        eprintln!("mid-layout brute-force check: {compared} positions compared");
    }

    #[test]
    fn alternate_keys_lie_outside_the_key_span() {
        // A base or alternate key can equal another position's base or
        // alternate key only if an alternate constant (or their XOR) is a sum of
        // point, side-to-move and pass keys. Over GF(2) they are not: every
        // collision involving an alternate slot is then a 64-bit accident, and
        // the second hash still rejects it.
        let mut gens: Vec<u64> = (0..30)
            .flat_map(|i| {
                [Color::Black, Color::White].map(|c| Board::<5, 6, 1, 30>::sym_zobrist(0, i, c))
            })
            .collect();
        gens.extend([STM_BLACK ^ STM_WHITE, PASS_KEY]);
        let rank = |vs: &[u64]| {
            let mut basis = [0u64; 64];
            let mut r = 0;
            for &v0 in vs {
                let mut v = v0;
                for bit in (0..64).rev() {
                    if (v >> bit) & 1 == 0 {
                        continue;
                    }
                    if basis[bit] == 0 {
                        basis[bit] = v;
                        r += 1;
                        break;
                    }
                    v ^= basis[bit];
                }
            }
            r
        };
        assert_eq!(rank(&gens), 62);
        for extra in [ALT_KEYS[0], ALT_KEYS[1], ALT_KEYS[0] ^ ALT_KEYS[1]] {
            let mut with = gens.clone();
            with.push(extra);
            assert_eq!(rank(&with), 63, "{extra:#x} is in the key span");
        }
    }

    #[test]
    fn position_keys_are_exact_to_49_points() {
        // Above 30 points the base key alone is not exact. A proof entry is
        // identified by its base key and its second hash together, a 128-bit
        // linear key. It is exact when the 2N point generators (base and second
        // key of each point and colour), the side-to-move generator and the pass
        // generator are linearly independent as 128-bit vectors. Alternate slots
        // change only the base key, so their constants (and their XOR) must lie
        // outside that span too. Keys depend only on the point index: 49 points
        // (7x7, the largest supported board) cover every board.
        let pair = |a: u64, b: u64| u128::from(a) | (u128::from(b) << 64);
        let mut gens: Vec<u128> = (0..49)
            .flat_map(|i| {
                [(Color::Black, 0), (Color::White, 1)]
                    .map(|(c, ci)| pair(Board::<7, 7, 1, 49>::sym_zobrist(0, i, c), k2_point(i, ci)))
            })
            .collect();
        gens.push(pair(STM_BLACK ^ STM_WHITE, K2_STM_BLACK ^ K2_STM_WHITE));
        gens.push(pair(PASS_KEY, K2_PASS));
        let rank = |vs: &[u128]| {
            let mut basis = [0u128; 128];
            let mut r = 0;
            for &v0 in vs {
                let mut v = v0;
                for bit in (0..128).rev() {
                    if (v >> bit) & 1 == 0 {
                        continue;
                    }
                    if basis[bit] == 0 {
                        basis[bit] = v;
                        r += 1;
                        break;
                    }
                    v ^= basis[bit];
                }
            }
            r
        };
        assert_eq!(rank(&gens), 100);
        for extra in [ALT_KEYS[0], ALT_KEYS[1], ALT_KEYS[0] ^ ALT_KEYS[1]] {
            let mut with = gens.clone();
            with.push(u128::from(extra));
            assert_eq!(rank(&with), 101, "{extra:#x} is in the 128-bit key span");
        }
    }

    #[test]
    fn shared_table_agrees_with_fresh_oracle_on_random_games() {
        let runs = [
            shared_table_check::<1, 5, 1, 5>(1, 40, 24, &[-5, -3, -1, 1]),
            shared_table_check::<1, 6, 1, 6>(2, 20, 24, &[-1, 1, 3]),
            shared_table_check::<2, 3, 1, 6>(3, 20, 16, &[-1, 1, 3]),
            shared_table_check::<3, 3, 1, 9>(4, 8, 14, &[5, 13, 17]),
        ];
        let compared: u64 = runs.iter().map(|r| r.0).sum();
        let rejects: u64 = runs.iter().map(|r| r.1).sum();
        assert!(compared > 1000, "only {compared} positions compared");
        eprintln!("shared-table check: {compared} positions compared, {rejects} frame rejects");
    }
}
