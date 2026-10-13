//! The program the page shows (everything it runs) and the pictures:
//! every picture's journey down the levels, and one picture large.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, Journey, LEVELS, N, PICTURES, SIDE};

/// What the page runs, all of it: the head, then the run's lines, the
/// 800 numbers of noise elided.
pub fn program() -> String {
    let t: Vec<String> = tail(&[0.0; PICTURES * N])
        .lines()
        .map(|l| match l {
            l if l.starts_with("E := ") => format!("E := {PICTURES} {N} r_eshape ...   # {} Gaussian numbers, the browser's", PICTURES * N),
            l => l.to_string(),
        })
        .collect();
    format!("{}# -- each run: eight pictures of noise, ten steps down, every level kept --\n{}\n", head(), t.join("\n"))
}

/// The ten steps.
pub fn range(src: &str) -> Range {
    between(src, "(_, _, L) := ", "(_, _, L) := ")
}

pub fn source() -> Html {
    let src = program();
    block(&src, range(&src))
}

/// The journey's cells and gaps, in pixels; the large picture's cell.
pub const CELL: usize = 4;
pub const GAP: usize = 4;
pub const BIG: usize = 12;
/// The journey: PICTURES rows of LEVELS + 1 frames.
pub const JW: usize = (LEVELS + 1) * (SIDE * CELL + GAP) - GAP;
pub const JH: usize = PICTURES * (SIDE * CELL + GAP) - GAP;
/// The large picture: LEVELS + 1 frames in a row.
pub const BW: usize = (LEVELS + 1) * (SIDE * BIG + GAP) - GAP;
pub const BH: usize = SIDE * BIG;

const ORANGE: Rgb = [255, 122, 0];

fn put(px: &mut [u8], cols: usize, r: usize, c: usize, rgb: Rgb) {
    let i = 4 * (r * cols + c);
    if i + 3 <= px.len() {
        px[i..i + 3].copy_from_slice(&rgb);
    }
}

/// A pixel's gray: -1 white, 1 black, between in between.
fn gray(v: f64) -> Rgb {
    let g = (127.5 * (1.0 - v.clamp(-1.0, 1.0))) as u8;
    [g, g, g]
}

/// One picture drawn at (top, left) with cells of `cell` pixels.
fn frame(px: &mut [u8], cols: usize, picture: &[f64], top: usize, left: usize, cell: usize) {
    for (i, &v) in picture.iter().enumerate() {
        let (r0, c0) = ((i / SIDE) * cell, (i % SIDE) * cell);
        for r in 0..cell {
            for c in 0..cell {
                put(px, cols, top + r0 + r, left + c0 + c, gray(v));
            }
        }
    }
}

/// Every picture's journey, a row each, the noise at the left and the
/// clean picture at the right; the picked row marked in orange.
pub fn journey(j: &Journey, pick: usize) -> Vec<u8> {
    let mut px = vec![255u8; 4 * JW * JH];
    for i in 0..PICTURES {
        let top = i * (SIDE * CELL + GAP);
        for l in 0..=LEVELS {
            frame(&mut px, JW, j.picture(l, i), top, l * (SIDE * CELL + GAP), CELL);
        }
        if i == pick {
            for r in 0..SIDE * CELL {
                for c in 0..2 {
                    put(&mut px, JW, top + r, c, ORANGE);
                    put(&mut px, JW, top + r, JW - 1 - c, ORANGE);
                }
            }
        }
    }
    px
}

/// The picked picture's journey, large.
pub fn large(j: &Journey, pick: usize) -> Vec<u8> {
    let mut px = vec![255u8; 4 * BW * BH];
    for l in 0..=LEVELS {
        frame(&mut px, BW, j.picture(l, pick), 0, l * (SIDE * BIG + GAP), BIG);
    }
    px
}
