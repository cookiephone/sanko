//! Checkpoints of a running solve: snapshots of the transposition table that a
//! later run on the same problem continues from (see
//! [`crate::dfpn::solve_dfpn_resumable`]).
//!
//! df-pn keeps its whole search state in the table. The workers' stacks are
//! rebuilt from the root, so a checkpoint is the table's two parts plus a
//! header naming the build and every setting that decides what a stored proof
//! means. A checkpoint is resumed only by the same build (source fingerprint),
//! on the same problem, with the same table sizes. The static cache is a cache
//! and is not saved.
//!
//! A checkpoint can also seed a run on a later position of the same game (its
//! moves extended by more moves), which is how a proof is split into parts:
//! every stored proof is reused there only through the transfer test on the
//! seeded run's own path, as within one run.
//!
//! And it can seed a run at another komi. A proof's leaves (game ends, static
//! verdicts, closed regions) compare areas with the komi, so a win for Black
//! stays a win at any lower komi and a win for White at any higher one. The
//! seeded run keeps only those proofs, and only if the earlier run trusted them
//! (it was not pruned for the other side); unproven numbers are kept as they
//! only steer the search.
//!
//! ## File format
//!
//! - Bytes `0..HEADER_BYTES`: UTF-8 text, the line `sanko-checkpoint 1`, then
//!   `key = value` lines, then `end`, padded with newlines.
//! - The unproven buckets, then the proof buckets, as little-endian 64-bit
//!   words, in chunks of whole buckets of at most `CHUNK_BYTES` bytes.
//! - One 64-bit checksum per chunk, in chunk order.
//!
//! A snapshot is written to `PATH.tmp` and renamed over `PATH` once complete,
//! so a crash while writing leaves the previous snapshot intact.

use std::fmt;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use crate::pntt::PnTable;

/// Size of the text header at the start of a checkpoint.
const HEADER_BYTES: u64 = 1 << 16;
/// Largest chunk of table data (whole buckets): the unit of checksums and of
/// parallel I/O.
const CHUNK_BYTES: u64 = 64 << 20;
const MAGIC: &str = "sanko-checkpoint 1";

/// Header keys that must agree for a checkpoint to be resumed: the build,
/// everything that decides what a stored proof means, and the table sizes.
/// The moves, the komi and the pruning are checked separately: the run's moves
/// must extend the checkpoint's, and another komi keeps only the proofs that
/// stay valid at it (see the module docs).
const MUST_MATCH: [&str; 12] = [
    "build",
    "board",
    "keep_tempo",
    "exact_only",
    "no_static",
    "region_solver",
    "region_max",
    "region_interior",
    "atari_dead",
    "layout",
    "unproven_buckets",
    "proof_buckets",
];

/// Checkpoint settings of [`crate::dfpn::solve_dfpn_resumable`].
#[derive(Clone, Debug, Default)]
pub struct CheckpointOptions {
    /// Write snapshots here, every `every` and at `stop_after`. An existing
    /// file is refused unless it is also the `resume` file.
    pub write: Option<PathBuf>,
    /// Continue from the snapshot at this path: the same position, or seed a
    /// run on a later position of the same game (see the module docs).
    pub resume: Option<PathBuf>,
    /// Time between snapshots (with `write`).
    pub every: Option<Duration>,
    /// Stop after this much wall time, with a last snapshot if `write` is set.
    pub stop_after: Option<Duration>,
    /// Also write a snapshot when the search ends with a result (to seed runs
    /// at other komi or of later positions, as a value search does).
    pub at_end: bool,
    /// The solver build (version and source fingerprint). A checkpoint is
    /// resumed only by the build that wrote it.
    pub build: String,
}

/// Why a checkpoint could not be written or resumed.
#[derive(Debug)]
pub enum CheckpointError {
    /// A file operation failed.
    Io { path: PathBuf, error: io::Error },
    /// The file is not a complete checkpoint, or its data is damaged.
    Format { path: PathBuf, reason: String },
    /// The checkpoint belongs to another build, problem or table size, or a
    /// fresh run would overwrite an existing checkpoint.
    Refused(String),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Format { path, reason } => write!(f, "{}: {reason}", path.display()),
            Self::Refused(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// The text header of a checkpoint: `key = value` pairs in file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointHeader {
    pub fields: Vec<(String, String)>,
}

impl CheckpointHeader {
    /// The value of `key`, if present.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Set `key` to `value`, in place if present, else at the end.
    pub(crate) fn set(&mut self, key: &str, value: impl fmt::Display) {
        let value = value.to_string();
        if let Some(f) = self.fields.iter_mut().find(|(k, _)| k == key) {
            f.1 = value;
        } else {
            self.fields.push((key.to_string(), value));
        }
    }

    /// The value of `key` as a number, 0 if absent.
    pub(crate) fn number(&self, key: &str) -> u64 {
        self.get(key).and_then(|v| v.parse().ok()).unwrap_or(0)
    }

    fn text(&self) -> String {
        let mut t = format!("{MAGIC}\n");
        for (k, v) in &self.fields {
            t.push_str(k);
            t.push_str(" = ");
            t.push_str(v);
            t.push('\n');
        }
        t.push_str("end\n");
        t
    }

    fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines();
        if lines.next()? != MAGIC {
            return None;
        }
        let mut fields = Vec::new();
        for line in lines {
            if line == "end" {
                return Some(Self { fields });
            }
            let (k, v) = line.split_once(" = ")?;
            fields.push((k.to_string(), v.to_string()));
        }
        None
    }

    /// The settings that must agree (see `MUST_MATCH`) in which the checkpoint
    /// `self` differs from the run `run`, one line each.
    pub(crate) fn mismatches(&self, run: &Self) -> Vec<String> {
        MUST_MATCH
            .iter()
            .filter_map(|&k| {
                let (a, b) = (self.get(k), run.get(k));
                (a != b).then(|| format!("{k}: {} in the checkpoint, {} in this run", a.unwrap_or("-"), b.unwrap_or("-")))
            })
            .collect()
    }
}

impl fmt::Display for CheckpointHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (k, v) in &self.fields {
            writeln!(f, "{k} = {v}")?;
        }
        Ok(())
    }
}

/// A run of whole buckets of one table part, and where it lies in the file.
#[derive(Clone, Copy, Debug)]
struct Chunk {
    part: usize,
    first: usize,
    buckets: usize,
    offset: u64,
}

/// The chunks of `tt` in file order, and the offset of the checksums after them.
fn plan(tt: &PnTable) -> (Vec<Chunk>, u64) {
    let mut chunks = Vec::new();
    let mut offset = HEADER_BYTES;
    for (part, (buckets, words)) in tt.parts().into_iter().enumerate() {
        let bucket_bytes = 8 * words as u64;
        let per_chunk = (CHUNK_BYTES / bucket_bytes).max(1) as usize;
        let mut first = 0;
        while first < buckets {
            let n = per_chunk.min(buckets - first);
            chunks.push(Chunk { part, first, buckets: n, offset });
            offset += n as u64 * bucket_bytes;
            first += n;
        }
    }
    (chunks, offset)
}

/// Checksum of a chunk's words (detects damaged or truncated data).
fn chunk_sum(words: &[u64]) -> u64 {
    words.iter().fold(0x243F_6A88_85A3_08D3, |h, &w| (h.rotate_left(23) ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[cfg(unix)]
fn write_at(file: &File, buf: &[u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}

#[cfg(windows)]
fn write_at(file: &File, mut buf: &[u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_write(buf, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        buf = &buf[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut std::mem::take(&mut buf)[n..];
        offset += n as u64;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn write_at(_: &File, _: &[u8], _: u64) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "checkpoints need positional file I/O"))
}

#[cfg(not(any(unix, windows)))]
fn read_at(_: &File, _: &mut [u8], _: u64) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "checkpoints need positional file I/O"))
}

/// Run `work` on every chunk with `threads` threads (each with its own word
/// and byte buffers) and collect the checksums it returns, or the first error.
fn parallel_chunks<F>(chunks: &[Chunk], threads: usize, work: F) -> io::Result<Vec<u64>>
where
    F: Fn(usize, Chunk, &mut Vec<u64>, &mut Vec<u8>) -> io::Result<u64> + Sync,
{
    let sums: Vec<AtomicU64> = chunks.iter().map(|_| AtomicU64::new(0)).collect();
    let next = AtomicUsize::new(0);
    let failed: Mutex<Option<io::Error>> = Mutex::new(None);
    std::thread::scope(|s| {
        for _ in 0..threads.clamp(1, chunks.len().max(1)) {
            s.spawn(|| {
                let (mut words, mut bytes) = (Vec::new(), Vec::new());
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= chunks.len() {
                        break;
                    }
                    match work(i, chunks[i], &mut words, &mut bytes) {
                        Ok(sum) => sums[i].store(sum, Ordering::Relaxed),
                        Err(e) => {
                            failed.lock().expect("error lock").get_or_insert(e);
                            next.store(chunks.len(), Ordering::Relaxed);
                            break;
                        }
                    }
                }
            });
        }
    });
    match failed.into_inner().expect("error lock") {
        Some(e) => Err(e),
        None => Ok(sums.into_iter().map(AtomicU64::into_inner).collect()),
    }
}

/// `path` with `.tmp` appended.
fn tmp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Write `tt` with `header` to `path`, through `path.tmp`, with `threads` I/O
/// threads. The table must not change meanwhile (the search is paused).
pub(crate) fn write_snapshot(tt: &PnTable, header: &CheckpointHeader, path: &Path, threads: usize) -> Result<(), CheckpointError> {
    let tmp = tmp_path(path);
    let io_err = |error: io::Error| CheckpointError::Io { path: tmp.clone(), error };
    let (chunks, sums_at) = plan(tt);
    let file = File::create(&tmp).map_err(io_err)?;
    file.set_len(sums_at + 8 * chunks.len() as u64).map_err(io_err)?;
    let sums = parallel_chunks(&chunks, threads, |_, c, words, bytes| {
        words.clear();
        tt.read_buckets(c.part, c.first..c.first + c.buckets, words);
        bytes.clear();
        bytes.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        write_at(&file, bytes, c.offset)?;
        Ok(chunk_sum(words))
    })
    .map_err(io_err)?;
    let tail: Vec<u8> = sums.iter().flat_map(|s| s.to_le_bytes()).collect();
    write_at(&file, &tail, sums_at).map_err(io_err)?;
    let mut h = header.clone();
    h.set("chunk_bytes", CHUNK_BYTES);
    h.set("chunks", chunks.len());
    h.set("checksums_offset", sums_at);
    let mut text = h.text().into_bytes();
    if text.len() as u64 > HEADER_BYTES {
        return Err(CheckpointError::Format { path: tmp.clone(), reason: "header too long".to_string() });
    }
    text.resize(HEADER_BYTES as usize, b'\n');
    write_at(&file, &text, 0).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    drop(file);
    std::fs::rename(&tmp, path).map_err(|error| CheckpointError::Io { path: path.to_path_buf(), error })
}

/// Read the checkpoint at `path`, whose header is `header`, into `tt`, which
/// was made with the table sizes the header names. Every chunk's checksum is
/// verified.
pub(crate) fn load_snapshot(tt: &PnTable, header: &CheckpointHeader, path: &Path, threads: usize) -> Result<(), CheckpointError> {
    let io_err = |error: io::Error| CheckpointError::Io { path: path.to_path_buf(), error };
    let bad = |reason: String| CheckpointError::Format { path: path.to_path_buf(), reason };
    let (chunks, sums_at) = plan(tt);
    let size = sums_at + 8 * chunks.len() as u64;
    if header.number("chunk_bytes") != CHUNK_BYTES
        || header.number("chunks") != chunks.len() as u64
        || header.number("checksums_offset") != sums_at
    {
        return Err(bad("its data layout does not match the table sizes".to_string()));
    }
    let file = File::open(path).map_err(io_err)?;
    let len = file.metadata().map_err(io_err)?.len();
    if len != size {
        return Err(bad(format!("{len} bytes, expected {size}")));
    }
    let mut tail = vec![0u8; 8 * chunks.len()];
    read_at(&file, &mut tail, sums_at).map_err(io_err)?;
    let sums: Vec<u64> = tail.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)).collect();
    parallel_chunks(&chunks, threads, |i, c, words, bytes| {
        let per_bucket = tt.parts()[c.part].1;
        bytes.resize(8 * c.buckets * per_bucket, 0);
        read_at(&file, bytes, c.offset)?;
        words.clear();
        words.extend(bytes.as_chunks::<8>().0.iter().map(|b| u64::from_le_bytes(*b)));
        let sum = chunk_sum(words);
        if sum != sums[i] {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("checksum mismatch in chunk {i} (byte {})", c.offset)));
        }
        tt.write_buckets(c.part, c.first, words);
        Ok(sum)
    })
    .map_err(|e| if e.kind() == io::ErrorKind::InvalidData { bad(e.to_string()) } else { io_err(e) })?;
    Ok(())
}

/// Read the header of the checkpoint at `path`.
pub fn read_checkpoint_header(path: &Path) -> Result<CheckpointHeader, CheckpointError> {
    let io_err = |error: io::Error| CheckpointError::Io { path: path.to_path_buf(), error };
    let file = File::open(path).map_err(io_err)?;
    let len = file.metadata().map_err(io_err)?.len();
    let mut buf = vec![0u8; len.min(HEADER_BYTES) as usize];
    read_at(&file, &mut buf, 0).map_err(io_err)?;
    CheckpointHeader::parse(&String::from_utf8_lossy(&buf))
        .ok_or_else(|| CheckpointError::Format { path: path.to_path_buf(), reason: "not a complete sanko checkpoint".to_string() })
}

/// Whether `a` and `b` name the same existing file.
pub(crate) fn same_file(a: &Path, b: &Path) -> bool {
    matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y)
}
