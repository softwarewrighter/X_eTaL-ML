//! Trains the ternary-net demo's network offline and writes its weights
//! into `../ternary-net.xtl`, between `# -- the weights` and
//! `# -- end of the weights`.
//!
//! The task: which of three spiral arms a point (x, y) in -1..1 belongs
//! to. The network: 2 -> 16 -> 16 -> 3, ReLU between layers. Two sets of
//! weights are trained:
//!
//! - `fp`: in full precision (Adam, softmax cross-entropy);
//! - `qa`: fine-tuned from `fp` for ternary weights, BitNet b1.58 style:
//!   the forward pass uses each layer's weights ternarized (threshold
//!   0.5 of the mean size, scale the mean size of the kept weights), the
//!   gradient updates the full-precision weights behind them (the
//!   straight-through estimator).
//!
//! Deterministic: the same weights every run. `just ternary-train`.

use std::f64::consts::PI;

const DIMS: [(usize, usize); 3] = [(2, 16), (16, 16), (16, 3)];
const THRESHOLD: f64 = 0.5;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-12), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}

/// `per` points on each of three spiral arms, with noise.
fn spiral(rng: &mut Rng, per: usize) -> Vec<([f64; 2], usize)> {
    let mut out = Vec::new();
    for c in 0..3 {
        for _ in 0..per {
            let t = rng.next();
            let r = 0.12 + 0.85 * t;
            let a = c as f64 * 2.0 * PI / 3.0 + 3.6 * t + 0.22 * rng.normal();
            out.push(([r * a.cos(), r * a.sin()], c));
        }
    }
    out
}

#[derive(Clone)]
struct Net {
    /// Per layer: weights [input][output] flattened, and biases.
    w: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        let w = DIMS.iter().map(|&(i, o)| (0..i * o).map(|_| rng.normal() * (2.0 / i as f64).sqrt()).collect()).collect();
        let b = DIMS.iter().map(|&(_, o)| vec![0.0; o]).collect();
        Net { w, b }
    }

    /// The weights the forward pass uses: as they are, or ternarized.
    fn effective(&self, ternary: bool) -> Vec<Vec<f64>> {
        self.w.iter().map(|w| if ternary { ternarize(w, THRESHOLD) } else { w.clone() }).collect()
    }

    /// Logits for each point, and the activations each layer saw.
    fn forward(&self, w: &[Vec<f64>], x: &[f64]) -> Vec<Vec<f64>> {
        let mut acts = vec![x.to_vec()];
        for (l, &(ni, no)) in DIMS.iter().enumerate() {
            let a = acts.last().unwrap();
            let mut z = self.b[l].clone();
            for i in 0..ni {
                for o in 0..no {
                    z[o] += a[i] * w[l][i * no + o];
                }
            }
            if l < 2 {
                z.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            acts.push(z);
        }
        acts
    }

    fn accuracy(&self, ternary: bool, data: &[([f64; 2], usize)]) -> f64 {
        let w = self.effective(ternary);
        let ok = data.iter().filter(|(x, c)| argmax(self.forward(&w, x).last().unwrap()) == *c).count();
        ok as f64 / data.len() as f64
    }

    /// Mean cross-entropy gradients over `data` with respect to the
    /// effective weights (passed straight through to the stored ones).
    fn gradients(&self, ternary: bool, data: &[([f64; 2], usize)]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let w = self.effective(ternary);
        let mut gw: Vec<Vec<f64>> = self.w.iter().map(|w| vec![0.0; w.len()]).collect();
        let mut gb: Vec<Vec<f64>> = self.b.iter().map(|b| vec![0.0; b.len()]).collect();
        for (x, c) in data {
            let acts = self.forward(&w, x);
            let out = acts.last().unwrap();
            let m = out.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = out.iter().map(|v| (v - m).exp()).collect();
            let s: f64 = e.iter().sum();
            let mut d: Vec<f64> = e.iter().map(|v| v / s).collect();
            d[*c] -= 1.0;
            for l in (0..3).rev() {
                let (ni, no) = DIMS[l];
                let a = &acts[l];
                for o in 0..no {
                    gb[l][o] += d[o];
                    for i in 0..ni {
                        gw[l][i * no + o] += a[i] * d[o];
                    }
                }
                if l > 0 {
                    let mut dn = vec![0.0; ni];
                    for i in 0..ni {
                        if a[i] > 0.0 {
                            dn[i] = (0..no).map(|o| w[l][i * no + o] * d[o]).sum();
                        }
                    }
                    d = dn;
                }
            }
        }
        let n = data.len() as f64;
        gw.iter_mut().flatten().for_each(|g| *g /= n);
        gb.iter_mut().flatten().for_each(|g| *g /= n);
        (gw, gb)
    }

    fn train(&mut self, ternary: bool, data: &[([f64; 2], usize)], epochs: usize, lr: f64) {
        let mut m: Vec<Vec<f64>> = self.w.iter().chain(&self.b).map(|v| vec![0.0; v.len()]).collect();
        let mut v = m.clone();
        let (b1, b2) = (0.9f64, 0.999f64);
        for t in 1..=epochs {
            let (gw, gb) = self.gradients(ternary, data);
            let params = self.w.iter_mut().chain(self.b.iter_mut());
            for (k, (p, g)) in params.zip(gw.iter().chain(&gb)).enumerate() {
                for j in 0..p.len() {
                    m[k][j] = b1 * m[k][j] + (1.0 - b1) * g[j];
                    v[k][j] = b2 * v[k][j] + (1.0 - b2) * g[j] * g[j];
                    let mh = m[k][j] / (1.0 - b1.powi(t as i32));
                    let vh = v[k][j] / (1.0 - b2.powi(t as i32));
                    p[j] -= lr * mh / (vh.sqrt() + 1e-8);
                }
            }
        }
    }

    /// Each weight and bias rounded to the decimals written out.
    fn rounded(&self) -> Self {
        let r = |v: &Vec<f64>| v.iter().map(|x| (x * 1e5).round() / 1e5).collect();
        Net { w: self.w.iter().map(r).collect(), b: self.b.iter().map(r).collect() }
    }

    /// The model as the X_eTaL program holds it: 17 x 3 x 16, for each
    /// row (16 inputs padded with 0, then the bias), layer and output.
    fn cube(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(17 * 3 * 16);
        for r in 0..17 {
            for (l, &(ni, no)) in DIMS.iter().enumerate() {
                for o in 0..16 {
                    out.push(match (r, o < no) {
                        (16, true) => self.b[l][o],
                        (r, true) if r < ni => self.w[l][r * no + o],
                        _ => 0.0,
                    });
                }
            }
        }
        out
    }
}

/// `w` ternarized: -1, 0 or +1 (kept where the size passes `t` times the
/// mean size), times the mean size of the kept weights.
fn ternarize(w: &[f64], t: f64) -> Vec<f64> {
    let g = w.iter().map(|x| x.abs()).sum::<f64>() / w.len() as f64;
    let q: Vec<f64> = w.iter().map(|&x| if x > t * g { 1.0 } else if x < -t * g { -1.0 } else { 0.0 }).collect();
    let kept = q.iter().filter(|q| **q != 0.0).count().max(1) as f64;
    let a = w.iter().zip(&q).map(|(x, q)| x.abs() * q.abs()).sum::<f64>() / kept;
    q.iter().map(|q| a * q).collect()
}

fn argmax(v: &[f64]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

fn num(x: f64) -> String {
    let s = format!("{x:.5}");
    let s = s.trim_end_matches('0');
    let s = if s.ends_with('.') { format!("{s}0") } else { s.to_string() };
    if s == "-0.0" { "0.0".into() } else { s }
}

fn strand(v: &[f64]) -> String {
    v.iter().map(|&x| num(x)).collect::<Vec<_>>().join(" ")
}

/// A long strand broken into lines of about 12 numbers, each line a
/// continuation inside parentheses (X_eTaL statements are one line, so
/// the program binds pieces and joins them with c_at).
fn bind(name: &str, shape: &str, v: &[f64]) -> String {
    let chunks: Vec<String> = v.chunks(48).map(strand).collect();
    let mut s = String::new();
    for (k, c) in chunks.iter().enumerate() {
        if k == 0 {
            s += &format!("{name} := {c}\n");
        } else {
            s += &format!("{name} := {name} c_at {c}\n");
        }
    }
    if !shape.is_empty() {
        s += &format!("{name} := ({shape}) r_eshape {name}\n");
    }
    s
}

fn main() {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let train = spiral(&mut rng, 100);
    let test = spiral(&mut rng, 40);
    let mut fp = Net::new(&mut rng);
    fp.train(false, &train, 3000, 0.01);
    let fp = fp.rounded();
    let mut qa = fp.clone();
    qa.train(true, &train, 3000, 0.005);
    let qa = qa.rounded();
    eprintln!("fp: train {:.3} test {:.3}; ternarized: test {:.3}", fp.accuracy(false, &train), fp.accuracy(false, &test), fp.accuracy(true, &test));
    eprintln!("qa: test as ternary {:.3}; in full precision {:.3}", qa.accuracy(true, &test), qa.accuracy(false, &test));

    let px: Vec<f64> = test.iter().map(|(x, _)| x[0]).collect();
    let py: Vec<f64> = test.iter().map(|(x, _)| x[1]).collect();
    let pl: Vec<f64> = test.iter().map(|(_, c)| (c + 1) as f64).collect();
    let section = format!(
        "# -- the weights (written by train/: just ternary-train) ------\n\
         # fp: trained in full precision; qa: fine-tuned for ternary\n\
         # weights. Each is 17 x 3 x 16: for each row (16 inputs, padded\n\
         # with 0, then the bias), layer and output (padded with 0).\n\
         {}{}# The test points: x, y, and the arm (1, 2 or 3).\n{}{}{}",
        bind("fp", "17 c_at 3 c_at 16", &fp.cube()),
        bind("qa", "17 c_at 3 c_at 16", &qa.cube()),
        bind("px", "", &px),
        bind("py", "", &py),
        bind("pl", "", &pl),
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../ternary-net.xtl");
    let src = std::fs::read_to_string(path).expect("read ternary-net.xtl");
    let (a, b) = (src.find("# -- the weights").expect("no weights marker"), src.find("# -- end of the weights").expect("no end marker"));
    std::fs::write(path, format!("{}{}{}", &src[..a], section, &src[b..])).expect("write ternary-net.xtl");
    eprintln!("wrote the weights into {path}");
}
