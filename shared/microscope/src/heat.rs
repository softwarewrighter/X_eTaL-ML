//! Small matrices as heat maps (from the LoRA explorer, shared at
//! X_eTaL-fine-tuning's ask FM5): diverging colors about 0 as the 3D
//! stage view colors its cubes (`color::DIVERGE`: red positive, blue
//! negative, near white about 0), a scale given so that maps side by
//! side compare, each cell's value on hover. Pages link
//! `microscope.css` for `figure.heat`.

use yew::prelude::*;

use crate::color::{ramp, DIVERGE};

/// A matrix read back: its rows, its columns, its items by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub items: Vec<f64>,
}

/// A vector as a one-column matrix.
pub fn column(items: &[f64]) -> Matrix {
    Matrix { rows: items.len(), cols: 1, items: items.to_vec() }
}

/// The largest size in some matrices: the scale that makes them compare.
pub fn largest(ms: &[&Matrix]) -> f64 {
    ms.iter().flat_map(|m| m.items.iter()).fold(0.0, |a: f64, v| a.max(v.abs()))
}

/// Cell size of a heat map, in pixels.
pub const CELL: f64 = 22.0;

/// The color of `v` against `scale` (a size of `scale` or more is the
/// full red or blue), as CSS.
pub fn fill(v: f64, scale: f64) -> String {
    let c = ramp(&DIVERGE, 0.5 + 0.5 * if scale > 0.0 { (v / scale).clamp(-1.0, 1.0) } else { 0.0 });
    format!("rgb({}, {}, {})", c[0], c[1], c[2])
}

/// `m` as a heat map under `label`, colored against `scale` (the largest
/// size shown beside it, `largest`); each cell's value on hover.
pub fn heat(m: &Matrix, scale: f64, label: &str) -> Html {
    let (w, h) = (m.cols as f64 * CELL, m.rows as f64 * CELL);
    let cells = m.items.iter().enumerate().map(|(k, v)| {
        let (y, x) = ((k / m.cols) as f64 * CELL, (k % m.cols) as f64 * CELL);
        html! { <rect x={x.to_string()} y={y.to_string()} width={CELL.to_string()} height={CELL.to_string()} fill={fill(*v, scale)} class="cell"><title>{format!("{v:.4}")}</title></rect> }
    });
    html! {
        <figure class="heat">
            <svg width={w.to_string()} height={h.to_string()} viewBox={format!("0 0 {w} {h}")} role="img" aria-label={label.to_string()}>
                { for cells }
                <rect x="0" y="0" width={w.to_string()} height={h.to_string()} class="frame" />
            </svg>
            <figcaption>{label}</figcaption>
        </figure>
    }
}
