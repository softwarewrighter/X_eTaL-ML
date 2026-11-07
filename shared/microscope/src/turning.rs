//! One turning view in three dimensions for any page (X_eTaL-manifolds'
//! ask GM3): points (a color, a label), segments (a path, a vector) and
//! wire lines (a surface's meridians and parallels), turned by yaw and
//! pitch from a pointer drag (a mouse, a pen or a finger), the arrow
//! keys, or the page itself; the angle is the page's state, so a preset
//! can name a view. The page's X_eTaL program computes the geometry;
//! this draws it and computes nothing a page shows.
//!
//! The projection, stated: each point is divided by the scale (a typical
//! radius, `typical`), pulled in to 1.25 when farther (so a few outliers
//! do not shrink the rest), turned by the yaw about the vertical axis
//! and then by the pitch about the horizontal one, and put in a mild
//! perspective, k = 1 / (1.6 - 0.35 z): the screen point is the middle
//! plus 0.72 size x k (x, -y), and z its depth (nearer is larger; the
//! nearest are drawn last). It is the embedding explorer's, from where
//! it began.

use yew::prelude::*;

/// How a view is turned: the yaw about the vertical axis and the pitch
/// about the horizontal one, in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Angle {
    pub yaw: f64,
    pub pitch: f64,
}

impl Default for Angle {
    fn default() -> Self {
        Angle { yaw: 0.6, pitch: 0.35 }
    }
}

/// The pitch stops short of straight up or down.
pub const PITCH: f64 = 1.5;

/// The angle after a drag of (dx, dy) pixels: 0.01 radian a pixel.
pub fn turn(a: Angle, dx: f64, dy: f64) -> Angle {
    Angle { yaw: a.yaw + 0.01 * dx, pitch: (a.pitch + 0.01 * dy).clamp(-PITCH, PITCH) }
}

/// The angle after an arrow key (`ArrowLeft` .. `ArrowDown`, a tenth of
/// a radian each), or None for any other key.
pub fn key(a: Angle, key: &str) -> Option<Angle> {
    match key {
        "ArrowLeft" => Some(turn(a, -10.0, 0.0)),
        "ArrowRight" => Some(turn(a, 10.0, 0.0)),
        "ArrowUp" => Some(turn(a, 0.0, -10.0)),
        "ArrowDown" => Some(turn(a, 0.0, 10.0)),
        _ => None,
    }
}

/// A typical radius of the points `xyz` (x y z, three numbers each): the
/// 85th percentile of their distances from the origin, times 1.15, so
/// a few outliers do not shrink the rest.
pub fn typical(xyz: &[f64]) -> f64 {
    let mut r: Vec<f64> = xyz.chunks(3).map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
    r.sort_by(f64::total_cmp);
    r.get(r.len() * 85 / 100).copied().unwrap_or(1.0).max(1e-12) * 1.15
}

/// The screen point of p in a size x size picture, turned by `a`,
/// divided by `scale`: (x, y, depth), the depth larger when nearer.
pub fn project(p: [f64; 3], a: Angle, scale: f64, size: f64) -> (f64, f64, f64) {
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() / scale;
    let c = if r > 1.25 { 1.25 / r } else { 1.0 };
    let (x, y, z) = (c * p[0] / scale, c * p[1] / scale, c * p[2] / scale);
    let (cy, sy, cp, sp) = (a.yaw.cos(), a.yaw.sin(), a.pitch.cos(), a.pitch.sin());
    let (x1, z1) = (x * cy + z * sy, -x * sy + z * cy);
    let (y2, z2) = (y * cp - z1 * sp, y * sp + z1 * cp);
    let k = 1.0 / (1.6 - 0.35 * z2);
    (size / 2.0 + 0.72 * size * x1 * k, size / 2.0 - 0.72 * size * y2 * k, z2)
}

/// A point to draw: where, its color (CSS), its label ("" for none).
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub at: [f64; 3],
    pub color: String,
    pub label: String,
}

/// A segment: from, to, its color, and whether it ends in an arrowhead
/// (a vector).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub color: String,
    pub arrow: bool,
}

/// What a view shows: points, segments, and wire lines (each a path of
/// points: a meridian, a parallel, a curve), the wires in one color.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub points: Vec<Point>,
    pub segments: Vec<Segment>,
    pub wires: Vec<Vec<[f64; 3]>>,
}

impl Scene {
    /// Every coordinate in the scene, for `typical`.
    pub fn coordinates(&self) -> Vec<f64> {
        let mut v: Vec<f64> = self.points.iter().flat_map(|p| p.at).collect();
        v.extend(self.segments.iter().flat_map(|s| s.from.into_iter().chain(s.to)));
        v.extend(self.wires.iter().flatten().flatten());
        v
    }
}

/// One thing on the screen, in the order to draw it.
#[derive(Clone, Debug, PartialEq)]
pub enum Mark {
    /// The i-th point at (x, y), its depth.
    Point { i: usize, x: f64, y: f64, depth: f64 },
    /// The i-th segment from (x1, y1) to (x2, y2), its mean depth.
    Segment { i: usize, x1: f64, y1: f64, x2: f64, y2: f64, depth: f64 },
    /// A wire's screen path.
    Wire { path: Vec<(f64, f64)> },
}

/// The scene's marks, turned and projected: the wires first (behind),
/// then points and segments farthest first, so the nearest are on top.
pub fn marks(scene: &Scene, a: Angle, scale: f64, size: f64) -> Vec<Mark> {
    let at = |p: [f64; 3]| project(p, a, scale, size);
    let mut out: Vec<Mark> = scene.wires.iter().map(|w| Mark::Wire { path: w.iter().map(|&p| { let (x, y, _) = at(p); (x, y) }).collect() }).collect();
    let mut solid: Vec<Mark> = scene.points.iter().enumerate().map(|(i, p)| { let (x, y, depth) = at(p.at); Mark::Point { i, x, y, depth } }).collect();
    solid.extend(scene.segments.iter().enumerate().map(|(i, s)| {
        let ((x1, y1, d1), (x2, y2, d2)) = (at(s.from), at(s.to));
        Mark::Segment { i, x1, y1, x2, y2, depth: 0.5 * (d1 + d2) }
    }));
    let depth = |m: &Mark| match m {
        Mark::Point { depth, .. } | Mark::Segment { depth, .. } => *depth,
        Mark::Wire { .. } => f64::NEG_INFINITY,
    };
    solid.sort_by(|p, q| depth(p).total_cmp(&depth(q)));
    out.extend(solid);
    out
}

#[derive(Properties, PartialEq)]
pub struct TurningProps {
    pub scene: Scene,
    /// The angle, the page's state; `turned` gives the new one.
    pub angle: Angle,
    pub turned: Callback<Angle>,
    /// What the picture shows, for a screen reader.
    pub label: AttrValue,
    /// The picture's side, in its own units (it scales to its box).
    #[prop_or(520.0)]
    pub size: f64,
    /// The scale, or None for the scene's typical radius.
    #[prop_or_default]
    pub scale: Option<f64>,
    /// Called with a point's index when it is clicked or tapped.
    #[prop_or_default]
    pub picked: Option<Callback<usize>>,
}

/// The view: an SVG turned by a drag (mouse, pen or finger) and the
/// arrow keys once focused; a point is picked by a click or a tap.
#[function_component(Turning)]
pub fn turning(p: &TurningProps) -> Html {
    let from = use_mut_ref(|| None::<(f64, f64)>);
    let size = p.size;
    let scale = p.scale.unwrap_or_else(|| typical(&p.scene.coordinates()));
    let down = { let from = from.clone(); Callback::from(move |e: PointerEvent| *from.borrow_mut() = Some((e.client_x() as f64, e.client_y() as f64))) };
    let mv = {
        let (from, turned, a) = (from.clone(), p.turned.clone(), p.angle);
        Callback::from(move |e: PointerEvent| {
            let at = (e.client_x() as f64, e.client_y() as f64);
            let last = *from.borrow();
            if let Some((x0, y0)) = last {
                *from.borrow_mut() = Some(at);
                turned.emit(turn(a, at.0 - x0, at.1 - y0));
            }
        })
    };
    let up = { let from = from.clone(); Callback::from(move |_: PointerEvent| *from.borrow_mut() = None) };
    let keys = {
        let (turned, a) = (p.turned.clone(), p.angle);
        Callback::from(move |e: KeyboardEvent| {
            if let Some(n) = key(a, &e.key()) {
                e.prevent_default();
                turned.emit(n);
            }
        })
    };
    let f = |v: f64| format!("{v:.1}");
    let drawn = marks(&p.scene, p.angle, scale, size).into_iter().map(|m| match m {
        Mark::Wire { path } => {
            let pts: Vec<String> = path.iter().map(|(x, y)| format!("{},{}", f(*x), f(*y))).collect();
            html! { <polyline class="wire" points={pts.join(" ")} /> }
        }
        Mark::Segment { i, x1, y1, x2, y2, .. } => {
            let s = &p.scene.segments[i];
            html! { <line class={classes!("seg", s.arrow.then_some("vector"))} x1={f(x1)} y1={f(y1)} x2={f(x2)} y2={f(y2)} stroke={s.color.clone()} /> }
        }
        Mark::Point { i, x, y, depth } => {
            let pt = &p.scene.points[i];
            let pick = p.picked.clone().map(|c| Callback::from(move |_: MouseEvent| c.emit(i)));
            html! { <g class="pt" onclick={pick}>
                <circle cx={f(x)} cy={f(y)} r={f((4.0 + 1.5 * depth).max(2.0))} fill={pt.color.clone()} />
                if !pt.label.is_empty() { <text x={f(x + 6.0)} y={f(y + 4.0)} font-size={f(10.0 + 3.0 * depth)} fill={pt.color.clone()}>{pt.label.clone()}</text> }
            </g> }
        }
    });
    html! {
        <svg class="turning" viewBox={format!("0 0 {size} {size}")} width="100%" role="img" aria-label={p.label.clone()} tabindex="0"
            onpointerdown={down} onpointermove={mv} onpointerup={up.clone()} onpointerleave={up.clone()} onpointercancel={up} onkeydown={keys}>
            <defs><marker id="turning-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke" /></marker></defs>
            { for drawn }
        </svg>
    }
}
