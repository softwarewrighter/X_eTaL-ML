//! Arrays to RGBA pixels, in the demos' shared palettes.

pub type Rgb = [u8; 3];

fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let f = |i: usize| (a[i] as f64 + (b[i] as f64 - a[i] as f64) * t).round() as u8;
    [f(0), f(1), f(2)]
}

/// A color from `stops` spread evenly over 0..1.
pub fn ramp(stops: &[Rgb], t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let i = (t.floor() as usize).min(stops.len() - 2);
    mix(stops[i], stops[i + 1], t - i as f64)
}

/// Dark navy through violet and orange to cream.
pub const GLOW: [Rgb; 5] = [[12, 10, 40], [88, 44, 160], [177, 151, 252], [255, 179, 102], [255, 244, 214]];
/// Blue, white, red.
pub const DIVERGE: [Rgb; 3] = [[37, 99, 235], [245, 245, 245], [220, 38, 38]];

/// RGBA pixels, one per value.
pub fn pixels(values: &[f64], color: impl Fn(f64) -> Rgb) -> Vec<u8> {
    values.iter().flat_map(|&v| { let [r, g, b] = color(v); [r, g, b, 255] }).collect()
}

/// Values from lo (dark) to hi (bright).
pub fn field(values: &[f64], lo: f64, hi: f64) -> Vec<u8> {
    pixels(values, |v| ramp(&GLOW, (v - lo) / (hi - lo)))
}

/// Non-negative values scaled by the largest.
pub fn scaled(values: &[f64]) -> Vec<u8> {
    let m = values.iter().fold(1e-12_f64, |m, &v| m.max(v));
    field(values, 0.0, m)
}

/// Signed values: blue below 0, red above, scaled by the largest size.
pub fn signed(values: &[f64]) -> Vec<u8> {
    let m = values.iter().fold(1e-12_f64, |m, v| m.max(v.abs()));
    pixels(values, |v| ramp(&DIVERGE, 0.5 + v / (2.0 * m)))
}

/// A 0 / 1 mask: 1 dark, 0 light violet.
pub fn mask(values: &[f64]) -> Vec<u8> {
    pixels(values, |v| if v == 1.0 { [24, 20, 48] } else { [229, 219, 255] })
}
