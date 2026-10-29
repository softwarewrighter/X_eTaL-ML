//! A stage in 3D: which earlier stages it reads (Flow's rule), and a
//! value drawn as a block of cubes, a cube an item colored by its value,
//! turned by yaw and pitch. Pure, tested natively; app.rs draws it.

/// The cubes drawn along an axis at most; a longer axis is drawn this
/// long, its true extent in the label.
pub const CAP: usize = 12;

/// The picture is SIZE by SIZE.
pub const SIZE: f64 = 240.0;

/// The earlier stages stage `k` (from 1, the data first) reads, by
/// Flow's rule: the previous value when it names `it` or names no
/// earlier stage (a function, applied to it); each earlier stage whose
/// label it names (inside an equation's strings, a name followed by
/// `[`, the target's string left out). `labels` is (stage, label) for
/// every labeled value. In order, each once.
pub fn inputs(text: &str, k: usize, labels: &[(usize, String)]) -> Vec<usize> {
    if k <= 1 {
        return Vec::new();
    }
    // The stage's own label (`$x`, `$(a, b)`) and shape are not reads.
    let body = text.find('$').map_or(text, |i| &text[..i]);
    let words = names(body);
    let mut out: Vec<usize> = Vec::new();
    let mut named = false;
    for w in &words {
        if let Some(&(s, _)) = labels.iter().find(|(s, l)| *s < k && l == w) {
            named = true;
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    if words.iter().any(|w| w == "it") || !named {
        if !out.contains(&(k - 1)) {
            out.push(k - 1);
        }
    }
    out.sort();
    out
}

/// The names a stage's text uses: outside strings every name; inside a
/// string a name followed by `[` (an array an equation reads) and `it`;
/// a string followed by `e:` (an equation's target) left out.
fn names(text: &str) -> Vec<String> {
    let b = text.as_bytes();
    let word = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b':';
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'"' {
            let close = text[i + 1..].find('"').map_or(b.len(), |j| i + 1 + j);
            let inner = &text[i + 1..close.min(b.len())];
            let target = text.get(close + 1..).is_some_and(|r| r.trim_start().starts_with("e:"));
            if !target {
                let ib = inner.as_bytes();
                let mut j = 0;
                while j < ib.len() {
                    if word(ib[j]) {
                        let s = j;
                        while j < ib.len() && word(ib[j]) {
                            j += 1;
                        }
                        let w = &inner[s..j];
                        if w == "it" || ib.get(j) == Some(&b'[') {
                            out.push(w.to_string());
                        }
                    } else {
                        j += 1;
                    }
                }
            }
            i = close + 1;
        } else if word(b[i]) {
            let s = i;
            while i < b.len() && word(b[i]) {
                i += 1;
            }
            out.push(text[s..i].to_string());
        } else {
            i += 1;
        }
    }
    out
}

/// A shape as the block drawn: (slices, rows, columns), the leading
/// axes of a rank above 3 folded into the slices.
pub fn block(shape: &[usize]) -> (usize, usize, usize) {
    match shape {
        [] => (1, 1, 1),
        [c] => (1, 1, *c),
        [r, c] => (1, *r, *c),
        _ => {
            let n = shape.len();
            (shape[..n - 2].iter().product(), shape[n - 2], shape[n - 1])
        }
    }
}

/// One visible face of a cube on the screen: its corners, the item it
/// shows (row-major), its light (1 a top face, less a side) and depth.
#[derive(Clone, Debug, PartialEq)]
pub struct Face {
    pub pts: [(f64, f64); 4],
    pub item: usize,
    pub light: f64,
    pub depth: f64,
}

/// The block on the screen: the faces to draw in order (the farthest
/// first), the outline of the whole block (drawn extents), and where
/// each axis' extent goes (columns, rows, slices).
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    pub faces: Vec<Face>,
    pub edges: Vec<((f64, f64), (f64, f64))>,
    pub labels: Vec<(f64, f64, String)>,
}

/// A value of `shape` drawn, its first `known` items as cubes (the rest
/// of the block outlined only), turned by yaw and pitch, fitted to SIZE.
pub fn picture(shape: &[usize], known: usize, yaw: f64, pitch: f64) -> Picture {
    let (s, r, c) = block(shape);
    let (ds, dr, dc) = (s.min(CAP), r.min(CAP), c.min(CAP));
    let (cy, sy, cp, sp) = (yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin());
    // Columns to the right, rows down, slices away; centered.
    let (mx, my, mz) = (dc as f64 / 2.0, -(dr as f64) / 2.0, -(ds as f64) / 2.0);
    let turn = |x: f64, y: f64, z: f64| {
        let (x, y, z) = (x - mx, y - my, z - mz);
        let (x1, z1) = (x * cy + z * sy, -x * sy + z * cy);
        (x1, y * cp - z1 * sp, y * sp + z1 * cp)
    };
    let box_pts: Vec<(f64, f64, f64)> = (0..8).map(|i| {
        let x = if i & 1 == 1 { dc as f64 } else { 0.0 };
        let y = if i & 2 == 2 { -(dr as f64) } else { 0.0 };
        let z = if i & 4 == 4 { -(ds as f64) } else { 0.0 };
        (x, y, z)
    }).collect();
    let flat: Vec<(f64, f64)> = box_pts.iter().map(|&(x, y, z)| { let p = turn(x, y, z); (p.0, p.1) }).collect();
    let (x0, x1) = flat.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.0), b.max(p.0)));
    let (y0, y1) = flat.iter().fold((f64::MAX, f64::MIN), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let scale = 0.72 * SIZE / (x1 - x0).max(y1 - y0).max(1e-9);
    let screen = |p: (f64, f64, f64)| (SIZE / 2.0 + scale * (p.0 - (x0 + x1) / 2.0), SIZE / 2.0 - scale * (p.1 - (y0 + y1) / 2.0));
    // The six faces of a unit cube: the normal, then the corners.
    let sides: [((f64, f64, f64), [(f64, f64, f64); 4], f64); 6] = [
        ((0.0, 1.0, 0.0), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, -1.0), (0.0, 0.0, -1.0)], 1.0),
        ((0.0, -1.0, 0.0), [(0.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, -1.0, -1.0), (0.0, -1.0, -1.0)], 0.6),
        ((0.0, 0.0, 1.0), [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, -1.0, 0.0), (0.0, -1.0, 0.0)], 0.88),
        ((0.0, 0.0, -1.0), [(0.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, -1.0, -1.0), (0.0, -1.0, -1.0)], 0.6),
        ((1.0, 0.0, 0.0), [(1.0, 0.0, 0.0), (1.0, 0.0, -1.0), (1.0, -1.0, -1.0), (1.0, -1.0, 0.0)], 0.74),
        ((-1.0, 0.0, 0.0), [(0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, -1.0), (0.0, -1.0, 0.0)], 0.74),
    ];
    let toward = |n: (f64, f64, f64)| {
        let z1 = -n.0 * sy + n.2 * cy;
        n.1 * sp + z1 * cp
    };
    let seen: Vec<_> = sides.iter().filter(|(n, _, _)| toward(*n) > 1e-9).collect();
    let mut cubes: Vec<(f64, usize, (f64, f64, f64))> = Vec::new();
    for a in 0..ds {
        for i in 0..dr {
            for j in 0..dc {
                let item = (a * r + i) * c + j;
                if item < known {
                    let o = (j as f64, -(i as f64), -(a as f64));
                    cubes.push((turn(o.0 + 0.5, o.1 - 0.5, o.2 - 0.5).2, item, o));
                }
            }
        }
    }
    cubes.sort_by(|p, q| p.0.total_cmp(&q.0));
    let gap = 0.08;
    let faces = cubes
        .iter()
        .flat_map(|&(depth, item, o)| {
            seen.iter().map(move |(_, pts, light)| (depth, item, o, pts, *light))
        })
        .map(|(depth, item, o, pts, light)| {
            let shrink = |v: f64| gap + v * (1.0 - 2.0 * gap);
            let p = |k: usize| screen(turn(o.0 + shrink(pts[k].0), o.1 - shrink(-pts[k].1), o.2 - shrink(-pts[k].2)));
            Face { pts: [p(0), p(1), p(2), p(3)], item, light, depth }
        })
        .collect();
    let edges = (0..8usize)
        .flat_map(|i| [1usize, 2, 4].into_iter().filter(move |b| i & b == 0).map(move |b| (i, i | b)))
        .map(|(i, j)| (screen(turn(box_pts[i].0, box_pts[i].1, box_pts[i].2)), screen(turn(box_pts[j].0, box_pts[j].1, box_pts[j].2))))
        .collect();
    let extent = |n: usize, d: usize| if n > d { format!("{n} ({d} drawn)") } else { n.to_string() };
    // Each axis' label beside the outermost of its four edges on the
    // screen, out from the middle of the picture.
    let (fc, fr, fs) = (dc as f64, -(dr as f64), -(ds as f64));
    let side = |ends: [((f64, f64, f64), (f64, f64, f64)); 4], text: String| {
        let (x, y) = ends
            .iter()
            .map(|&(a, b)| {
                let (p, q) = (screen(turn(a.0, a.1, a.2)), screen(turn(b.0, b.1, b.2)));
                ((p.0 + q.0) / 2.0, (p.1 + q.1) / 2.0)
            })
            .max_by(|p, q| ((p.0 - SIZE / 2.0).powi(2) + (p.1 - SIZE / 2.0).powi(2)).total_cmp(&((q.0 - SIZE / 2.0).powi(2) + (q.1 - SIZE / 2.0).powi(2))))
            .unwrap();
        let (dx, dy) = (x - SIZE / 2.0, y - SIZE / 2.0);
        let len = (dx * dx + dy * dy).sqrt().max(1e-9);
        (x + 13.0 * dx / len, y + 13.0 * dy / len, text)
    };
    let mut labels = Vec::new();
    if !shape.is_empty() {
        labels.push(side([((0.0, 0.0, 0.0), (fc, 0.0, 0.0)), ((0.0, fr, 0.0), (fc, fr, 0.0)), ((0.0, 0.0, fs), (fc, 0.0, fs)), ((0.0, fr, fs), (fc, fr, fs))], extent(c, dc)));
    }
    if shape.len() >= 2 {
        labels.push(side([((0.0, 0.0, 0.0), (0.0, fr, 0.0)), ((fc, 0.0, 0.0), (fc, fr, 0.0)), ((0.0, 0.0, fs), (0.0, fr, fs)), ((fc, 0.0, fs), (fc, fr, fs))], extent(r, dr)));
    }
    if shape.len() >= 3 {
        labels.push(side([((0.0, 0.0, 0.0), (0.0, 0.0, fs)), ((fc, 0.0, 0.0), (fc, 0.0, fs)), ((0.0, fr, 0.0), (0.0, fr, fs)), ((fc, fr, 0.0), (fc, fr, fs))], extent(s, ds)));
    }
    Picture { faces, edges, labels }
}
