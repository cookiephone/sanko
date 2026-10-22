//! The live panel of an interactive solve, on stderr: time, speed and the
//! root's proof numbers with their history, the root's moves on the board and
//! the fill of the tables, redrawn in place a few times a second. It needs a
//! colour terminal large enough for it, and falls back to the one-line status
//! otherwise. Plain output never shows it.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Mutex;

use anstyle::{AnsiColor, Style};
use sanko_core::Color;
use sanko_search::Progress;

use crate::ui::{self, ACCENT, DIM, GOOD, LABEL, LAST, paint};

/// What the panel knows about the solve, set before it starts.
pub(crate) struct Context {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// The root board, row-major from the top.
    pub(crate) cells: Vec<Option<Color>>,
    /// Bytes of the table's unproven part, its proof part and the static cache.
    pub(crate) bytes: [u64; 3],
}

struct State {
    ctx: Option<Context>,
    /// `(seconds, expansions, root phi, root delta)`, at most one per second.
    history: Vec<(f64, u64, u64, u64)>,
    /// Lines of the panel currently on the screen.
    drawn: usize,
    frame: usize,
}

static STATE: Mutex<State> = Mutex::new(State { ctx: None, history: Vec::new(), drawn: 0, frame: 0 });

const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// Points of the sparklines.
const SPARK: usize = 12;
/// Cells of a gauge bar.
const BAR: usize = 16;
const REFUTED: Style = AnsiColor::Red.on_default().dimmed();

/// Prepare the panel for a solve.
pub(crate) fn start(ctx: Context) {
    let mut st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    *st = State { ctx: Some(ctx), history: Vec::new(), drawn: 0, frame: 0 };
}

/// Progress callback of a solve.
pub(crate) fn progress(p: &Progress) {
    let Ok(mut st) = STATE.try_lock() else { return };
    if st.history.last().is_none_or(|h| p.elapsed - h.0 >= 1.0) {
        st.history.push((p.elapsed, p.nodes, p.root_phi, p.root_delta));
    }
    st.frame += 1;
    let colour = anstream::AutoStream::choice(&std::io::stderr()) != anstream::ColorChoice::Never;
    let size = terminal_size::terminal_size_of(std::io::stderr());
    let lines = match (colour, size, st.ctx.as_ref()) {
        (true, Some((terminal_size::Width(w), terminal_size::Height(h))), Some(ctx)) => {
            let lines = render(ctx, &st.history, p, usize::from(w), st.frame);
            (lines.len() + 2 <= usize::from(h)).then_some(lines)
        }
        _ => None,
    };
    if let Some(lines) = lines {
        draw(&mut st, &lines);
    } else {
        erase(&mut st);
        drop(st);
        ui::solve_progress(p);
    }
}

/// Remove the panel (or the one-line status).
pub(crate) fn clear() {
    let mut st = STATE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    erase(&mut st);
    st.ctx = None;
    drop(st);
    ui::clear_status();
}

fn draw(st: &mut State, lines: &[String]) {
    let mut out = String::new();
    if st.drawn > 0 {
        let _ = write!(out, "\x1b[{}A", st.drawn);
    }
    for line in lines {
        out.push_str("\r\x1b[2K");
        out.push_str(line);
        out.push('\n');
    }
    // A shorter panel than the last one: blank the rest and return.
    let extra = st.drawn.saturating_sub(lines.len());
    for _ in 0..extra {
        out.push_str("\r\x1b[2K\n");
    }
    if extra > 0 {
        let _ = write!(out, "\x1b[{extra}A");
    }
    st.drawn = lines.len();
    write_raw(&out);
}

fn erase(st: &mut State) {
    if st.drawn == 0 {
        return;
    }
    let n = st.drawn;
    write_raw(&format!("\x1b[{n}A{}\x1b[{n}A", "\r\x1b[2K\n".repeat(n)));
    st.drawn = 0;
}

fn write_raw(s: &str) {
    let mut err = anstream::stderr();
    let _ = err.write_all(s.as_bytes());
    let _ = err.flush();
}

fn render(ctx: &Context, history: &[(f64, u64, u64, u64)], p: &Progress, width: usize, frame: usize) -> Vec<String> {
    let rate = p.nodes as f64 / p.elapsed.max(1e-9);
    // All sparklines share one line: block characters on adjacent lines would
    // run into each other.
    let mut lines = vec![
        format!(
            "  {} {}  ·  {} expansions  ·  {}/s  ·  depth {}",
            paint(ACCENT, SPINNER[frame % SPINNER.len()]),
            paint(LABEL, ui::duration(p.elapsed)),
            ui::compact(p.nodes as f64),
            ui::compact(rate),
            p.max_ply
        ),
        String::new(),
        String::new(),
    ];
    // Fixed widths, so the line keeps its layout from the first frame: the
    // numbers right-aligned, the sparklines filling in from the right.
    let root = format!(
        "    root φ {} {}   δ {} {}",
        paint(LABEL, lpad(&ui::pn(p.root_phi), 6)),
        paint(ACCENT, lpad(&spark_log(&series(history, |h| h.2)), SPARK)),
        paint(LABEL, lpad(&ui::pn(p.root_delta), 6)),
        paint(ACCENT, lpad(&spark_log(&series(history, |h| h.3)), SPARK))
    );
    // The speed history only if it fits (the speed itself is on the first line).
    let with_speed = format!("{root}   speed {}", paint(ACCENT, lpad(&spark_linear(&rates(history)), SPARK)));
    lines[1] = if ui::visible_len(&with_speed) < width { with_speed } else { root };
    let (marks, pass) = marks(ctx, p);
    let board = board_lines(ctx, &marks);
    let gauges = gauge_lines(ctx, p);
    let board_width = 2 * ctx.cols + 6;
    let gauge_width = gauges.iter().map(|g| ui::visible_len(g)).max().unwrap_or(0);
    if width > board_width + 6 + gauge_width {
        for i in 0..board.len().max(gauges.len()) {
            let left = board.get(i).map_or(String::new(), Clone::clone);
            let pad = board_width.saturating_sub(ui::visible_len(&left));
            lines.push(format!("{left}{}      {}", " ".repeat(pad), gauges.get(i).map_or("", String::as_str)));
        }
    } else {
        lines.extend(board);
        lines.push(String::new());
        lines.extend(gauges.into_iter().map(|g| format!("    {g}")));
    }
    lines.push(String::new());
    let marks_legend = format!(
        "{} win  {} refuted  {} searched now",
        mark_glyph(Mark::Win),
        mark_glyph(Mark::Refuted),
        mark_glyph(Mark::Focus)
    );
    let open_legend = format!(
        "{} open, from most to least promising   pass {}",
        [Mark::Open(0), Mark::Open(1), Mark::Open(2)].map(mark_glyph).concat(),
        pass.map_or_else(|| paint(DIM, "·"), mark_glyph)
    );
    let legend = format!("    {marks_legend}  {open_legend}");
    if ui::visible_len(&legend) < width {
        lines.push(legend);
    } else {
        lines.push(format!("    {marks_legend}"));
        lines.push(format!("    {open_legend}"));
    }
    lines.into_iter().map(|l| fit(&l, width)).collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mark {
    Win,
    Refuted,
    /// The open move with the smallest disproof number: the search's choice.
    Focus,
    /// An open move, shade 0 (most promising) to 2.
    Open(u8),
}

fn mark_glyph(m: Mark) -> String {
    match m {
        Mark::Win => paint(GOOD, "★"),
        Mark::Refuted => paint(REFUTED, "✕"),
        Mark::Focus => paint(LAST, "◉"),
        // Three shapes, so the levels differ even where bold or dim text looks
        // like plain text (shaded blocks would merge across rows).
        Mark::Open(0) => paint(AnsiColor::BrightCyan.on_default().bold(), "◆"),
        Mark::Open(1) => paint(ACCENT, "◈"),
        Mark::Open(_) => paint(ACCENT.dimmed(), "◇"),
    }
}

/// The status of every point's move (and of the pass) at the root.
fn marks(ctx: &Context, p: &Progress) -> (Vec<Option<Mark>>, Option<Mark>) {
    let points = ctx.rows * ctx.cols;
    let open: Vec<u64> = p.root_moves.iter().filter(|m| m.1 != 0 && m.2 != 0).map(|m| m.2).collect();
    let (lo, hi) = (open.iter().copied().min().unwrap_or(1), open.iter().copied().max().unwrap_or(1));
    let (llo, lhi) = ((lo.max(1) as f64).ln(), (hi.max(1) as f64).ln());
    let mut focus_done = false;
    let mark = |phi: u64, delta: u64, focus_done: &mut bool| {
        if delta == 0 {
            Mark::Win
        } else if phi == 0 {
            Mark::Refuted
        } else if delta == lo && !*focus_done {
            *focus_done = true;
            Mark::Focus
        } else if lhi > llo {
            Mark::Open((((delta.max(1) as f64).ln() - llo) / (lhi - llo) * 2.999) as u8)
        } else {
            Mark::Open(0)
        }
    };
    let mut marks = vec![None; points];
    let mut pass = None;
    for &(code, phi, delta) in &p.root_moves {
        let m = mark(phi, delta, &mut focus_done);
        if code < points {
            marks[code] = Some(m);
        } else {
            pass = Some(m);
        }
    }
    // Moves merged with a symmetric one share its value.
    for i in 0..points {
        if marks[i].is_none()
            && ctx.cells[i].is_none()
            && let Some(m) = p.root_symmetries.iter().find_map(|map| map.get(i).and_then(|&j| marks.get(j).copied().flatten()))
        {
            marks[i] = Some(if m == Mark::Focus { Mark::Open(0) } else { m });
        }
    }
    (marks, pass)
}

fn board_lines(ctx: &Context, marks: &[Option<Mark>]) -> Vec<String> {
    let letters: String = (0..ctx.cols).map(|c| ui::column_letter(c).to_string()).collect::<Vec<_>>().join(" ");
    let mut lines = vec![format!("       {}", paint(DIM, letters))];
    for r in 0..ctx.rows {
        let cells: Vec<String> = (0..ctx.cols)
            .map(|c| {
                let i = r * ctx.cols + c;
                match (ctx.cells[i], marks[i]) {
                    (Some(colour), _) => ui::stone(colour).to_string(),
                    (None, Some(m)) => mark_glyph(m),
                    (None, None) => paint(DIM, "·"),
                }
            })
            .collect();
        lines.push(format!("    {} {}", paint(DIM, format!("{:>2}", ctx.rows - r)), cells.join(" ")));
    }
    lines
}

fn gauge_lines(ctx: &Context, p: &Progress) -> Vec<String> {
    let f = &p.table_fill;
    let gauge = |name: &str, used: f64, bytes: u64| {
        format!("{}  {}  {:>3.0}%  of {}", paint(LABEL, format!("{name:<6}")), bar(used), used * 100.0, ui::bytes(bytes))
    };
    vec![
        paint(DIM, "tables"),
        gauge("search", f.unproven, ctx.bytes[0]),
        gauge("proofs", f.proofs, ctx.bytes[1]),
        paint(DIM, format!("        {:.0}% of the proofs path-specific", f.path_specific * 100.0)),
        gauge("static", p.static_fill, ctx.bytes[2]),
    ]
}

/// A gauge of `BAR` cells drawn as a thin line (full-height blocks on adjacent
/// lines would merge), with half cells.
fn bar(fraction: f64) -> String {
    let halves = (fraction.clamp(0.0, 1.0) * (BAR * 2) as f64).round() as usize;
    let mut used = "━".repeat(halves / 2);
    if halves % 2 == 1 {
        used.push('╸');
    }
    let free = BAR - halves.div_ceil(2);
    format!("{}{}", paint(ACCENT, used), paint(DIM, "─".repeat(free)))
}

/// `s` padded with spaces on the left to `n` columns (one column per char).
fn lpad(s: &str, n: usize) -> String {
    format!("{}{s}", " ".repeat(n.saturating_sub(s.chars().count())))
}

/// The last value of each of up to `SPARK` equal stretches of the history.
fn series(history: &[(f64, u64, u64, u64)], value: impl Fn(&(f64, u64, u64, u64)) -> u64) -> Vec<u64> {
    stretches(history.len()).into_iter().map(|(_, end)| value(&history[end])).collect()
}

/// Expansions per second over each stretch of the history.
fn rates(history: &[(f64, u64, u64, u64)]) -> Vec<f64> {
    stretches(history.len())
        .into_iter()
        .map(|(start, end)| {
            let (a, b) = (&history[start], &history[end]);
            (b.1 - a.1) as f64 / (b.0 - a.0).max(1e-9)
        })
        .collect()
}

/// `(first, last)` sample indices of up to `SPARK` consecutive stretches, each
/// sharing its first sample with the previous one's last.
fn stretches(n: usize) -> Vec<(usize, usize)> {
    if n < 2 {
        return Vec::new();
    }
    let k = SPARK.min(n - 1);
    (0..k).map(|i| (i * (n - 1) / k, (i + 1) * (n - 1) / k)).collect()
}

fn spark_linear(values: &[f64]) -> String {
    let max = values.iter().copied().fold(0.0, f64::max);
    if max <= 0.0 {
        return String::new();
    }
    values.iter().map(|&v| BLOCKS[((v / max) * 7.0).round().clamp(0.0, 7.0) as usize]).collect()
}

fn spark_log(values: &[u64]) -> String {
    let logs: Vec<f64> = values.iter().map(|&v| (v.max(1) as f64).ln()).collect();
    let (lo, hi) = logs.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    logs.iter()
        .map(|&l| if hi > lo { BLOCKS[((l - lo) / (hi - lo) * 7.0).round() as usize] } else { BLOCKS[3] })
        .collect()
}

/// `line` cut to fit `width` columns (style codes kept, colours reset at a cut).
fn fit(line: &str, width: usize) -> String {
    let max = width.saturating_sub(1);
    if ui::visible_len(line) <= max {
        return line.to_string();
    }
    let mut out = String::new();
    let mut n = 0;
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            out.push(c);
            for e in chars.by_ref() {
                out.push(e);
                if e.is_ascii_alphabetic() {
                    break;
                }
            }
        } else if n < max {
            out.push(c);
            n += 1;
        }
    }
    out.push_str("\x1b[0m");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparklines_keep_their_width_while_the_history_grows() {
        let history: Vec<(f64, u64, u64, u64)> =
            (0..40u32).map(|i| (f64::from(i), 1000 * u64::from(i), 7 + u64::from(i), 99)).collect();
        for n in 0..history.len() {
            let h = &history[..n];
            assert_eq!(lpad(&spark_log(&series(h, |x| x.2)), SPARK).chars().count(), SPARK);
            assert_eq!(lpad(&spark_linear(&rates(h)), SPARK).chars().count(), SPARK);
        }
        assert_eq!(lpad(&ui::pn(12), 6), "    12");
        assert_eq!(lpad(&ui::pn(45_300), 6), "45.3 k");
    }
}
