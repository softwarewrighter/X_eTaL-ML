//! The program the page shows (everything it runs) and the map's pixels.

use microscope::color::Rgb;
use microscope::source::{between, block, Range};
use yew::Html;

use crate::micro::{head, tail, Start, SIDE};

/// What the page runs, all of it: the head, then a run's lines, the
/// centers it carries from run to run and the inertia history elided.
pub fn program(k: usize, start: Start, stepped: bool, runs: usize) -> String {
    let centers = vec![0.0; 2 * k];
    let earlier = vec![0.0; runs];
    let t: Vec<String> = tail(k, start, stepped.then_some(&centers[..]), &earlier)
        .lines()
        .map(|l| match l {
            l if l.starts_with("C := ") => format!("C := {k} 2 r_eshape ...   # the centers after the steps so far"),
            l if l.starts_with("inertia := (") => "inertia := (...) c_at C1 ml:i_nertia X   # the inertia after each step so far".to_string(),
            l => l.to_string(),
        })
        .collect();
    format!("{}# -- each run: one step from the centers the page holds --\n{}\n", head(), t.join("\n"))
}

/// The step (or the start).
pub fn range(src: &str) -> Range {
    between(src, "C1 := ", "C1 := ")
}

pub fn source(k: usize, start: Start, stepped: bool, runs: usize) -> Html {
    let src = program(k, start, stepped, runs);
    block(&src, range(&src))
}

const LIGHT: [Rgb; 8] = [[255, 224, 178], [200, 230, 201], [187, 222, 251], [248, 187, 208], [225, 190, 231], [255, 249, 196], [178, 235, 242], [215, 204, 200]];
const DARK: [Rgb; 8] = [[230, 81, 0], [27, 94, 32], [13, 71, 161], [173, 20, 87], [106, 27, 154], [190, 160, 0], [0, 131, 143], [93, 64, 55]];

/// The picture is PX by PX: the map's cells, the points, the trails, the centers.
pub const PX: usize = 240;

fn at(x: f64, y: f64) -> (i64, i64) {
    (((1.0 - y) / 2.0 * PX as f64) as i64, ((x + 1.0) / 2.0 * PX as f64) as i64)
}

fn put(px: &mut [u8], r: i64, c: i64, rgb: Rgb) {
    if (0..PX as i64).contains(&r) && (0..PX as i64).contains(&c) {
        let i = 4 * (r as usize * PX + c as usize);
        px[i..i + 3].copy_from_slice(&rgb);
    }
}

/// The map: each cell in its nearest center's light color; each point a
/// dot in its center's dark color; each center's path so far as small
/// gray dots; each center a cross, white on black.
pub fn map(map: &[usize], points: &[f64], points_in: &[usize], trail: &[Vec<f64>]) -> Vec<u8> {
    let cell = PX / SIDE;
    let mut px = vec![255u8; 4 * PX * PX];
    for r in 0..PX {
        for c in 0..PX {
            let m = map[(r / cell) * SIDE + c / cell];
            put(&mut px, r as i64, c as i64, LIGHT[(m.max(1) - 1) % 8]);
        }
    }
    for (p, &j) in points.chunks(2).zip(points_in) {
        let (r, c) = at(p[0], p[1]);
        for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            put(&mut px, r + dr, c + dc, DARK[(j.max(1) - 1) % 8]);
        }
    }
    for centers in trail.iter().take(trail.len().saturating_sub(1)) {
        for p in centers.chunks(2) {
            let (r, c) = at(p[0], p[1]);
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                put(&mut px, r + dr, c + dc, [90, 90, 90]);
            }
        }
    }
    if let Some(last) = trail.last() {
        for p in last.chunks(2) {
            let (r, c) = at(p[0], p[1]);
            for d in -5i64..=5 {
                for w in -1i64..=1 {
                    put(&mut px, r + d, c + w, [0, 0, 0]);
                    put(&mut px, r + w, c + d, [0, 0, 0]);
                }
            }
            for d in -4i64..=4 {
                put(&mut px, r + d, c, [255, 255, 255]);
                put(&mut px, r, c + d, [255, 255, 255]);
            }
        }
    }
    px
}
