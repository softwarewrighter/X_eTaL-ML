//! A matrix as a heat map a newcomer can read: X_eTaL-ML's
//! `microscope::heat` colors (red positive, blue negative, near white
//! about 0), with what it lacked for a lesson: a legend, a note saying
//! whether the scale is shared with the maps beside it, cells marked as
//! read or written by the step being shown, and an inspector that says
//! which matrix, row and column a cell is, its value and what it means.
//! The grid is one tab stop; the arrow keys move over its cells, and a
//! tap or a click picks one. Offered back to X_eTaL-ML (its FM5).

use web_sys::KeyboardEvent;
use yew::prelude::*;

pub use microscope::heat::{column, fill, largest, Matrix};

use crate::arithmetic::show;

/// Cell size, in pixels: larger than microscope's, for a finger.
pub const CELL: f64 = 30.0;

/// Room around the grid, so the outlines of marked cells are not cut off.
const PAD: f64 = 3.0;

/// What a step does to a cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mark {
    /// The step reads it.
    Read,
    /// The step writes it.
    Written,
}

/// What the inspector says of cell `k` (by rows, from 0) of `m`: which
/// it is, counted from 1 as the page's tables are, its value, and how
/// to read its color.
pub fn inspect(name: &str, m: &Matrix, k: usize) -> String {
    let (row, col, v) = (k / m.cols + 1, k % m.cols + 1, m.items[k]);
    let place = match (m.rows, m.cols) {
        (1, 1) => name.to_string(),
        (_, 1) => format!("{name}, number {row}"),
        (1, _) => format!("{name}, number {col}"),
        _ => format!("{name}, row {row}, column {col}"),
    };
    let color = if v > 0.0 { "positive, so red" } else if v < 0.0 { "negative, so blue" } else { "zero, so white" };
    format!("{place}: {} ({color})", show(v))
}

/// The legend's five stops, from the most negative to the most
/// positive value the scale shows: each value with its color.
pub fn legend(scale: f64) -> Vec<(f64, String)> {
    [-1.0, -0.5, 0.0, 0.5, 1.0].iter().map(|t| (t * scale, fill(t * scale, scale))).collect()
}

/// What the note under a legend says about its scale.
pub fn scale_note(shared: bool) -> &'static str {
    if shared {
        "The pictures in this group share one scale: the same color is the same number in each of them."
    } else {
        "This picture has a scale of its own: its colors cannot be compared with another picture's."
    }
}

/// Where the arrow keys take the picked cell of an `rows` x `cols`
/// grid, staying inside it; other keys leave it.
pub fn step(picked: usize, rows: usize, cols: usize, key: &str) -> usize {
    let (r, c) = (picked / cols, picked % cols);
    let (r, c) = match key {
        "ArrowLeft" => (r, c.saturating_sub(1)),
        "ArrowRight" => (r, (c + 1).min(cols - 1)),
        "ArrowUp" => (r.saturating_sub(1), c),
        "ArrowDown" => ((r + 1).min(rows - 1), c),
        "Home" => (r, 0),
        "End" => (r, cols - 1),
        _ => (r, c),
    };
    r * cols + c
}

/// The legend of a scale, drawn: five colors with their values, and the
/// note on whether the scale is shared. A group of maps sharing a scale
/// shows it once, under the group.
pub fn key(scale: f64, shared: bool) -> Html {
    let stops = legend(scale).into_iter().map(|(v, color)| html! { <span class="stop"><span class="swatch" style={format!("background: {color}")}></span>{ brief(v) }</span> });
    html! {
        <div class="heat-key">
            <div class="legend" aria-label="Color scale"><span class="lead">{"The colors, from the most negative number to the most positive:"}</span>{ for stops }</div>
            <p class="legend-note">{ scale_note(shared) }</p>
        </div>
    }
}

#[derive(Properties, PartialEq)]
pub struct HeatProps {
    /// The matrix's name in the text ("W").
    pub name: AttrValue,
    /// A caption: what the matrix is.
    pub caption: AttrValue,
    pub matrix: Matrix,
    /// The size the full red and blue stand for (`largest` of the maps
    /// shown together, or of this one alone).
    pub scale: f64,
    /// Whether that scale is shared with the maps beside this one.
    #[prop_or_default]
    pub shared: bool,
    /// Cells the step being shown reads or writes: (cell, mark).
    #[prop_or_default]
    pub marks: Vec<(usize, Mark)>,
    /// What a cell of this matrix means, said by the inspector.
    #[prop_or_default]
    pub meaning: AttrValue,
    /// Whether to draw the legend (once is enough for maps sharing a scale).
    #[prop_or(true)]
    pub legend: bool,
    /// Whether to write each value in its cell, as well as coloring it
    /// (for small maps a newcomer reads number by number).
    #[prop_or_default]
    pub numbers: bool,
    /// The side of a cell, in pixels.
    #[prop_or(CELL)]
    pub cell: f64,
    /// Not worked out yet (a later stage of a calculation being stepped
    /// through): the cells are drawn empty, with no value to read.
    #[prop_or_default]
    pub pending: bool,
    /// Whether to say, under the map, how to read a cell (once is
    /// enough for a group of maps).
    #[prop_or(true)]
    pub hint: bool,
    /// Told which cell was picked (by rows, from 0), so a page can show
    /// more about it (where the number comes from).
    #[prop_or_default]
    pub onpick: Callback<usize>,
    /// The whole map singled out (its symbol was chosen in a formula).
    #[prop_or_default]
    pub chosen: bool,
}

/// A value as written in a cell: two decimals at most.
pub fn brief(v: f64) -> String {
    show((v * 100.0).round() / 100.0)
}

#[function_component(Heat)]
pub fn heat(props: &HeatProps) -> Html {
    let picked = use_state(|| None::<usize>);
    let m = &props.matrix;
    let side = props.cell;
    let (w, h) = (m.cols as f64 * side, m.rows as f64 * side);
    // Marked cells come in one after another: the order they are worked out in.
    let mut order = 0usize;
    let cells = m.items.iter().enumerate().map(|(k, v)| {
        let (y, x) = ((k / m.cols) as f64 * side, (k % m.cols) as f64 * side);
        let mark = props.marks.iter().find(|(c, _)| *c == k).map(|(_, mk)| match mk {
            Mark::Read => "read",
            Mark::Written => "written",
        });
        let pick = { let (picked, onpick) = (picked.clone(), props.onpick.clone()); Callback::from(move |_| { picked.set(Some(k)); onpick.emit(k); }) };
        let delay = if mark.is_some() { order += 1; format!("animation-delay: {}ms", (order - 1) * 140) } else { String::new() };
        // Dark figures on a pale cell, light ones on a strong color.
        let strong = props.scale > 0.0 && (v / props.scale).abs() > 0.55;
        if props.pending {
            return html! { <g class="cellg pending"><rect x={x.to_string()} y={y.to_string()} width={side.to_string()} height={side.to_string()} class="cell pending" /><text x={(x + side / 2.0).to_string()} y={(y + side / 2.0).to_string()} class="num pending">{"?"}</text></g> };
        }
        html! {
            <g class={classes!("cellg", mark)} style={delay}>
                <rect x={x.to_string()} y={y.to_string()} width={side.to_string()} height={side.to_string()} fill={fill(*v, props.scale)}
                    class={classes!("cell", mark, (*picked == Some(k)).then_some("picked"))} onclick={pick}>
                    <title>{ inspect(&props.name, m, k) }</title>
                </rect>
                if props.numbers {
                    <text x={(x + side / 2.0).to_string()} y={(y + side / 2.0).to_string()} class={classes!("num", strong.then_some("strong"))}>{ brief(*v) }</text>
                }
            </g>
        }
    });
    let onkeydown = {
        let (picked, rows, cols, onpick, pending) = (picked.clone(), m.rows, m.cols, props.onpick.clone(), props.pending);
        Callback::from(move |e: KeyboardEvent| {
            let key = e.key();
            if !pending && ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].contains(&key.as_str()) {
                e.prevent_default();
                let to = step(picked.unwrap_or(0), rows, cols, &key);
                picked.set(Some(to));
                onpick.emit(to);
            }
        })
    };
    let onfocus = { let picked = picked.clone(); Callback::from(move |_| if picked.is_none() { picked.set(Some(0)) }) };
    html! {
        <figure class={classes!("heat", "lesson-heat", props.chosen.then_some("chosen"))}>
            <figcaption>{ props.caption.clone() }</figcaption>
            <svg width={(w + 2.0 * PAD).to_string()} height={(h + 2.0 * PAD).to_string()} viewBox={format!("{} {} {} {}", -PAD, -PAD, w + 2.0 * PAD, h + 2.0 * PAD)} role="img" tabindex="0"
                aria-label={format!("{}: a table of {} rows and {} columns. Arrow keys move over its cells.", props.caption, m.rows, m.cols)} {onkeydown} {onfocus}>
                { for cells }
                <rect x="0" y="0" width={w.to_string()} height={h.to_string()} class="frame" />
            </svg>
            <p class="inspector" role="status">{
                match *picked {
                    _ if props.pending => html! { <span class="hint">{"Not worked out yet."}</span> },
                    Some(k) => html! { <>{ inspect(&props.name, m, k) }{ if props.meaning.is_empty() { html! {} } else { html! { <>{". "}{ props.meaning.clone() }</> } } }</> },
                    None if !props.hint => html! {},
                    None => html! { <span class="hint">{"Click a cell, or use the arrow keys, to read it."}</span> },
                }
            }</p>
            if props.legend {
                { key(props.scale, props.shared) }
            }
        </figure>
    }
}
