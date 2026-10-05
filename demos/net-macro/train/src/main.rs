//! Trains the net-macro demo's networks, offline and deterministically.
//!
//! The task: which of three spiral arms a point (x, y) is on. The
//! networks are the ones net-macro.xtl writes with the macro: every
//! line `u:NAME := "SPEC" net:n_etwork< "W1 W2 ..."` is read from the
//! program, so the program is the one place a network is described.
//! Each is trained by Adam on softmax cross-entropy and its weights are
//! written as the macro's functions take them: data/W.txt per dense
//! layer, inputs + 1 rows by outputs columns, the bias the last row.
//! data/points.txt holds the test points, one per line: x, y, arm.
//! `just net-train`.

use std::fmt::Write;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2.0 * self.next().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * self.next()).cos()
    }
}

/// `per` points on each of three spiral arms, in about -1..1.
fn spiral(rng: &mut Rng, per: usize) -> Vec<([f64; 2], usize)> {
    let mut out = Vec::new();
    for arm in 0..3 {
        for i in 0..per {
            let t = (i as f64 + 0.5) / per as f64;
            let r = 0.1 + 0.9 * t;
            let a = arm as f64 * std::f64::consts::TAU / 3.0 + 5.5 * t + 0.10 * rng.normal();
            out.push(([r * a.cos(), r * a.sin()], arm));
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Act {
    None,
    Relu,
    Tanh,
    Sigmoid,
}

/// A dense layer: w is (inputs + 1) x outputs, the bias the last row.
struct Layer {
    ni: usize,
    no: usize,
    w: Vec<f64>,
    act: Act,
}

struct Net {
    layers: Vec<Layer>,
}

fn act(a: Act, v: f64) -> f64 {
    match a {
        Act::None => v,
        Act::Relu => v.max(0.0),
        Act::Tanh => v.tanh(),
        Act::Sigmoid => 1.0 / (1.0 + (-v).exp()),
    }
}

/// The derivative of the activation, from its output.
fn slope(a: Act, out: f64) -> f64 {
    match a {
        Act::None => 1.0,
        Act::Relu => (out > 0.0) as u8 as f64,
        Act::Tanh => 1.0 - out * out,
        Act::Sigmoid => out * (1.0 - out),
    }
}

impl Net {
    /// From a spec's words: sizes and activations (softmax ends it and is
    /// applied by the loss).
    fn new(spec: &str, rng: &mut Rng) -> Net {
        let mut layers: Vec<Layer> = Vec::new();
        let mut last = 0;
        for word in spec.split_whitespace() {
            match word.parse::<usize>() {
                Ok(n) if last == 0 => last = n,
                Ok(n) => {
                    let s = (2.0 / last as f64).sqrt();
                    let mut w: Vec<f64> = (0..last * n).map(|_| s * rng.normal()).collect();
                    w.extend(std::iter::repeat(0.0).take(n));
                    layers.push(Layer { ni: last, no: n, w, act: Act::None });
                    last = n;
                }
                Err(_) => {
                    let a = match word {
                        "relu" => Act::Relu,
                        "tanh" => Act::Tanh,
                        "sigmoid" => Act::Sigmoid,
                        "softmax" => Act::None,
                        other => panic!("the trainer does not know the activation {other}"),
                    };
                    if a != Act::None {
                        layers.last_mut().expect("an activation after a layer").act = a;
                    }
                }
            }
        }
        Net { layers }
    }

    /// Every layer's output for x (the last one before softmax).
    fn forward(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let mut outs: Vec<Vec<f64>> = vec![x.to_vec()];
        for l in &self.layers {
            let h = outs.last().unwrap();
            let o = (0..l.no).map(|j| act(l.act, (0..l.ni).map(|i| h[i] * l.w[i * l.no + j]).sum::<f64>() + l.w[l.ni * l.no + j])).collect();
            outs.push(o);
        }
        outs
    }

    fn predict(&self, x: &[f64]) -> usize {
        let o = self.forward(x).pop().unwrap();
        (0..o.len()).fold(0, |b, i| if o[i] > o[b] { i } else { b })
    }

    fn accuracy(&self, data: &[([f64; 2], usize)]) -> f64 {
        data.iter().filter(|(x, c)| self.predict(x) == *c).count() as f64 / data.len() as f64
    }

    /// Adam on the mean softmax cross-entropy over the whole set.
    fn train(&mut self, data: &[([f64; 2], usize)], steps: usize, lr: f64) {
        let sizes: Vec<usize> = self.layers.iter().map(|l| l.w.len()).collect();
        let mut m: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
        let mut v = m.clone();
        for step in 1..=steps {
            let mut g: Vec<Vec<f64>> = sizes.iter().map(|&n| vec![0.0; n]).collect();
            for (x, c) in data {
                let outs = self.forward(x);
                let z = outs.last().unwrap();
                let mx = z.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = z.iter().map(|t| (t - mx).exp()).collect();
                let s: f64 = e.iter().sum();
                let mut d: Vec<f64> = (0..z.len()).map(|j| e[j] / s - (j == *c) as u8 as f64).collect();
                for (k, l) in self.layers.iter().enumerate().rev() {
                    let (h, o) = (&outs[k], &outs[k + 1]);
                    let dz: Vec<f64> = (0..l.no).map(|j| d[j] * slope(l.act, o[j])).collect();
                    for j in 0..l.no {
                        for i in 0..l.ni {
                            g[k][i * l.no + j] += dz[j] * h[i];
                        }
                        g[k][l.ni * l.no + j] += dz[j];
                    }
                    d = (0..l.ni).map(|i| (0..l.no).map(|j| dz[j] * l.w[i * l.no + j]).sum()).collect();
                }
            }
            let n = data.len() as f64;
            for (k, l) in self.layers.iter_mut().enumerate() {
                for i in 0..l.w.len() {
                    let gi = g[k][i] / n;
                    m[k][i] = 0.9 * m[k][i] + 0.1 * gi;
                    v[k][i] = 0.999 * v[k][i] + 0.001 * gi * gi;
                    let (mh, vh) = (m[k][i] / (1.0 - 0.9f64.powi(step as i32)), v[k][i] / (1.0 - 0.999f64.powi(step as i32)));
                    l.w[i] -= lr * mh / (vh.sqrt() + 1e-8);
                }
            }
        }
    }
}

fn num(x: f64) -> String {
    let s = format!("{x:.5}");
    let s = s.trim_end_matches('0');
    let s = if s.ends_with('.') { format!("{s}0") } else { s.to_string() };
    if s == "-0.0" { "0.0".into() } else { s }
}

/// The lines `u:NAME := "SPEC" net:n_etwork< "W1 W2"` of the program:
/// (name, spec, weight names).
fn networks(program: &str) -> Vec<(String, String, Vec<String>)> {
    program
        .lines()
        .filter_map(|l| {
            let (name, rest) = l.strip_prefix("u:")?.split_once(" := \"")?;
            let (spec, rest) = rest.split_once("\" net:n_etwork< \"")?;
            let names = rest.strip_suffix('"')?;
            Some((name.to_string(), spec.to_string(), names.split_whitespace().map(str::to_string).collect()))
        })
        .collect()
}

fn main() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/..");
    let program = std::fs::read_to_string(format!("{root}/net-macro.xtl")).expect("read net-macro.xtl");
    let mut rng = Rng(0x5eed_0f_a_5b1ca1);
    let train = spiral(&mut rng, 100);
    let test = spiral(&mut rng, 40);
    std::fs::create_dir_all(format!("{root}/data")).expect("data/");
    let mut points = String::new();
    for (x, c) in &test {
        writeln!(points, "{} {} {}", num(x[0]), num(x[1]), c + 1).unwrap();
    }
    std::fs::write(format!("{root}/data/points.txt"), points).expect("write points");
    for (name, spec, names) in networks(&program) {
        let mut net = Net::new(&spec, &mut rng);
        assert_eq!(net.layers.len(), names.len(), "{name}: one weight array per dense layer");
        net.train(&train, 4000, 0.01);
        for l in &mut net.layers {
            for w in &mut l.w {
                *w = num(*w).parse().unwrap();
            }
        }
        eprintln!("{name} ({spec}): train {:.3}, test {:.3}", net.accuracy(&train), net.accuracy(&test));
        for (l, w) in net.layers.iter().zip(&names) {
            let text: String = l.w.chunks(l.no).map(|r| r.iter().map(|&x| num(x)).collect::<Vec<_>>().join(" ") + "\n").collect();
            std::fs::write(format!("{root}/data/{w}.txt"), text).expect("write weights");
        }
    }
    eprintln!("wrote data/points.txt and each layer's weights");
}
