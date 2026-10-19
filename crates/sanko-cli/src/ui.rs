//! Terminal presentation for interactive use: colours, Go coordinates, board
//! diagrams, a live status line and aligned statistics. Plain output (logs,
//! pipes, `--plain`) does not use its formatting, so its format stays stable
//! for scripts. Only error messages on stderr come from here in every mode.
//!
//! All output goes through `anstream`, which drops the colours when they are
//! turned off (`--color never`, `NO_COLOR`) or not supported.

use std::fmt::Display;
use std::io::Write;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use anstyle::{AnsiColor, Style};
use sanko_core::{Color, Move, Point, Position};

pub(crate) const TITLE: Style = AnsiColor::Cyan.on_default().bold();
pub(crate) const LABEL: Style = Style::new().bold();
pub(crate) const DIM: Style = Style::new().dimmed();
pub(crate) const GOOD: Style = AnsiColor::Green.on_default().bold();
pub(crate) const BAD: Style = AnsiColor::Red.on_default().bold();
pub(crate) const WARN: Style = AnsiColor::Yellow.on_default().bold();
pub(crate) const ACCENT: Style = AnsiColor::Cyan.on_default();
pub(crate) const LAST: Style = AnsiColor::Yellow.on_default().bold();

/// Column letters of Go coordinates (`I` is skipped, as on a real board).
const LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// `text` in `style`.
pub(crate) fn paint(style: Style, text: impl Display) -> String {
    format!("{style}{text}{style:#}")
}

/// Columns before a row's value: two spaces, an 11-column label, two spaces.
const VALUE_COLUMN: usize = 15;

/// One labelled line of a report: `  Label       value`. A value too long for
/// the terminal wraps at spaces, and its continuation lines start under the
/// value instead of at the left edge.
pub(crate) fn row(label: &str, value: impl Display) {
    row_lines(label, &[value.to_string()]);
}

/// A labelled entry of several lines, each starting under the value column.
pub(crate) fn row_lines(label: &str, lines: &[String]) {
    let indent = " ".repeat(VALUE_COLUMN);
    let width = value_width();
    for (i, line) in lines.iter().enumerate() {
        let head = if i == 0 { paint(LABEL, format!("{label:<11}")) } else { " ".repeat(11) };
        anstream::println!("  {head}  {}", wrap(line, width).join(&format!("\n{indent}")));
    }
}

/// Columns available for a value, or `None` when the terminal width is unknown.
fn value_width() -> Option<usize> {
    static WIDTH: OnceLock<Option<usize>> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        terminal_size::terminal_size().map(|(terminal_size::Width(w), _)| usize::from(w).saturating_sub(VALUE_COLUMN).max(24))
    })
}

/// Greedy word wrap of `text` to `width` visible columns (ANSI style codes take
/// no space). A word longer than a line keeps a line of its own.
fn wrap(text: &str, width: Option<usize>) -> Vec<String> {
    let Some(width) = width else { return vec![text.to_string()] };
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut used = 0;
    for word in text.split(' ') {
        let len = visible_len(word);
        if used > 0 && used + 1 + len > width {
            lines.push(std::mem::take(&mut line));
            used = 0;
        }
        if used > 0 {
            line.push(' ');
            used += 1;
        }
        line.push_str(word);
        used += len;
    }
    lines.push(line);
    lines
}

/// Characters of `s` that take a column, skipping ANSI escape sequences.
pub(crate) fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for e in chars.by_ref() {
                if e.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            n += 1;
        }
    }
    n
}

/// The title line of a command.
pub(crate) fn header(command: &str, version: &str, source: &str) {
    anstream::println!();
    anstream::println!(
        "  {}  {}   {}",
        paint(TITLE, "SANKO"),
        paint(LABEL, command),
        paint(DIM, format!("v{version} · source {}", &source[..source.len().min(12)]))
    );
    anstream::println!();
}

/// An error message (on stderr).
pub(crate) fn error(msg: &str) {
    anstream::eprintln!("{} {msg}", paint(BAD, "error:"));
}

/// `1234567` as `1,234,567`.
pub(crate) fn grouped(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A count with a metric suffix: `912`, `45.3 k`, `6.83 B`.
pub(crate) fn compact(x: f64) -> String {
    let (v, unit) = if x >= 1e12 {
        (x / 1e12, " T")
    } else if x >= 1e9 {
        (x / 1e9, " B")
    } else if x >= 1e6 {
        (x / 1e6, " M")
    } else if x >= 1e3 {
        (x / 1e3, " k")
    } else {
        return format!("{x:.0}");
    };
    if v >= 100.0 {
        format!("{v:.0}{unit}")
    } else if v >= 10.0 {
        format!("{v:.1}{unit}")
    } else {
        format!("{v:.2}{unit}")
    }
}

/// A duration in the most readable unit.
pub(crate) fn duration(secs: f64) -> String {
    let whole = secs as u64;
    if secs < 1.0 {
        format!("{:.0} ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{secs:.2} s")
    } else if secs < 3600.0 {
        format!("{} min {:02} s", whole / 60, whole % 60)
    } else {
        format!("{} h {:02} min", whole / 3600, (whole % 3600) / 60)
    }
}

/// A proof or disproof number (`∞` for a solved side).
pub(crate) fn pn(v: u64) -> String {
    if v >= u64::MAX / 2 { "∞".to_string() } else { compact(v as f64) }
}

/// Go coordinate of a point: column letter from the left, row number from the
/// bottom (`A1` is the bottom-left corner).
pub(crate) fn coord<const R: usize, const C: usize>(p: Point<R, C>) -> String {
    format!("{}{}", LETTERS[p.col()] as char, R - p.row())
}

/// A move in Go coordinates, or `pass`.
pub(crate) fn move_name<const R: usize, const C: usize>(m: Move<R, C>) -> String {
    match m {
        Move::Pass => "pass".to_string(),
        Move::Stone(p) => coord(p),
    }
}

/// Moves in Go coordinates, shortened to `max` moves with an ellipsis.
pub(crate) fn move_list<const R: usize, const C: usize>(moves: &[Move<R, C>], max: usize) -> String {
    let mut text: Vec<String> = moves.iter().take(max).map(|&m| move_name(m)).collect();
    if moves.len() > max {
        text.push("…".to_string());
    }
    text.join(" ")
}

/// A size in bytes with a binary unit: `448 GiB`, `1.5 MiB`.
pub(crate) fn bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut unit = 0;
    while v >= 1024.0 && unit + 1 < UNITS.len() {
        v /= 1024.0;
        unit += 1;
    }
    if v >= 100.0 || v.fract() == 0.0 { format!("{v:.0} {}", UNITS[unit]) } else { format!("{v:.1} {}", UNITS[unit]) }
}

/// The letter of column `c` in Go coordinates.
pub(crate) const fn column_letter(c: usize) -> char {
    LETTERS[c] as char
}

/// The point named by a Go coordinate (`C3`), if it is on the board.
pub(crate) fn parse_coord<const R: usize, const C: usize>(token: &str) -> Option<Point<R, C>> {
    let mut chars = token.chars();
    let letter = chars.next()?.to_ascii_uppercase();
    let col = LETTERS.iter().position(|&l| l as char == letter)?;
    let row_from_bottom: usize = chars.as_str().parse().ok()?;
    if row_from_bottom == 0 || row_from_bottom > R || col >= C {
        return None;
    }
    Point::new(R - row_from_bottom, col)
}

/// A board diagram with Go coordinates. `last` is highlighted, `marks` are
/// empty points drawn as a red cross.
pub(crate) fn board_lines<const R: usize, const C: usize, const L: usize, const N: usize>(
    pos: &Position<R, C, L, N>,
    last: Option<Point<R, C>>,
    marks: &[Point<R, C>],
) -> Vec<String> {
    let letters: Vec<String> = (0..C).map(|c| (LETTERS[c] as char).to_string()).collect();
    let mut lines = vec![format!("       {}", paint(DIM, letters.join(" ")))];
    for r in 0..R {
        let mut line = format!("    {} ", paint(DIM, format!("{:>2}", R - r)));
        for c in 0..C {
            let p = Point::<R, C>::new_unchecked(r, c);
            let (glyph, style) = match pos.board().at(p) {
                Some(Color::Black) => ("●", Style::new()),
                Some(Color::White) => ("○", Style::new()),
                None if marks.contains(&p) => ("✕", BAD),
                None => ("·", DIM),
            };
            let style = if Some(p) == last { LAST } else { style };
            line.push_str(&paint(style, glyph));
            if c + 1 < C {
                line.push(' ');
            }
        }
        lines.push(line);
    }
    lines
}

/// Print a board diagram.
pub(crate) fn board<const R: usize, const C: usize, const L: usize, const N: usize>(
    pos: &Position<R, C, L, N>,
    last: Option<Point<R, C>>,
    marks: &[Point<R, C>],
) {
    for line in board_lines(pos, last, marks) {
        anstream::println!("{line}");
    }
}

/// `●` for Black, `○` for White.
pub(crate) const fn stone(c: Color) -> &'static str {
    match c {
        Color::Black => "●",
        Color::White => "○",
    }
}

// ---------------------------------------------------------------------------
// Live status line (stderr)

static STATUS_LEN: AtomicUsize = AtomicUsize::new(0);
static SPIN: AtomicUsize = AtomicUsize::new(0);
static CLOCK: OnceLock<Instant> = OnceLock::new();
/// Milliseconds (since `CLOCK`) of the last redraw, plus one (0 = never).
static LAST_DRAW: AtomicU64 = AtomicU64::new(0);
const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Show `text` as the live status line on stderr (overwriting the last one).
/// Redraws at most about 12 times a second.
pub(crate) fn status(text: &str) {
    let now = CLOCK.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1;
    let last = LAST_DRAW.load(Ordering::Relaxed);
    if last != 0 && now < last + 80 {
        return;
    }
    LAST_DRAW.store(now, Ordering::Relaxed);
    let frame = SPINNER[SPIN.fetch_add(1, Ordering::Relaxed) % SPINNER.len()];
    let width = text.chars().count() + 4;
    let pad = STATUS_LEN.swap(width, Ordering::Relaxed).saturating_sub(width);
    let mut err = anstream::stderr();
    let _ = write!(err, "\r  {} {text}{}", paint(ACCENT, frame), " ".repeat(pad));
    let _ = err.flush();
}

/// Remove the live status line.
pub(crate) fn clear_status() {
    LAST_DRAW.store(0, Ordering::Relaxed);
    let width = STATUS_LEN.swap(0, Ordering::Relaxed);
    if width > 0 {
        let mut err = anstream::stderr();
        let _ = write!(err, "\r{}\r", " ".repeat(width));
        let _ = err.flush();
    }
}

/// Progress callback of a solve: elapsed time, expansions, rate, the root's
/// proof numbers and the deepest ply.
pub(crate) fn solve_progress(p: &sanko_search::Progress) {
    status(&format!(
        "{} · {} expansions · {}/s · root φ {} δ {} · depth {}",
        duration(p.elapsed),
        compact(p.nodes as f64),
        compact(p.nodes as f64 / p.elapsed.max(1e-9)),
        pn(p.root_phi),
        pn(p.root_delta),
        p.max_ply
    ));
}
