//! Trains the cnn-digits demo's tiny CNN on MNIST offline and writes
//! its weights, its test accuracy and ten sample test digits into
//! `../cnn-digits.xtl`, between `# -- the weights` and
//! `# -- end of the weights`.
//!
//! The network: a 28 x 28 picture -> 8 filters of 3 x 3 (valid, so 8 x
//! 26 x 26) -> ReLU -> 2 x 2 max-pooling (8 x 13 x 13) -> flattened
//! (1352, filter by filter, row by row) -> dense 10 -> softmax.
//! SGD with momentum, cross-entropy, 3 passes over the 60,000 training
//! digits, deterministic (a fixed shuffle). Needs `scripts/mnist.sh`
//! first; `just cnn-train` runs both.

const F: usize = 8;
const C: usize = 26;
const P: usize = 13;
const FLAT: usize = F * P * P;

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
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// An IDX file's payload after its header.
fn idx(path: &str, header: usize) -> Vec<u8> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e} (run scripts/mnist.sh first)"));
    b[header..].to_vec()
}

fn mnist(dir: &str, set: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let img = idx(&format!("{dir}/{set}-images-idx3-ubyte"), 16);
    let lab = idx(&format!("{dir}/{set}-labels-idx1-ubyte"), 8);
    let xs = img.chunks(784).map(|c| c.iter().map(|&p| p as f64 / 255.0).collect()).collect();
    (xs, lab.iter().map(|&l| l as usize).collect())
}

#[derive(Clone)]
struct Net {
    k: Vec<f64>,  // F x 3 x 3
    bc: Vec<f64>, // F
    w: Vec<f64>,  // FLAT x 10
    bd: Vec<f64>, // 10
}

struct Pass {
    conv: Vec<f64>, // F x C x C, after ReLU
    flat: Vec<f64>, // FLAT
    arg: Vec<usize>, // for each pooled value, the index in conv it came from
    out: Vec<f64>,  // 10 probabilities
}

impl Net {
    fn new(rng: &mut Rng) -> Self {
        Net {
            k: (0..F * 9).map(|_| rng.normal() * (2.0f64 / 9.0).sqrt()).collect(),
            bc: vec![0.0; F],
            w: (0..FLAT * 10).map(|_| rng.normal() * (1.0 / FLAT as f64).sqrt()).collect(),
            bd: vec![0.0; 10],
        }
    }

    fn forward(&self, x: &[f64]) -> Pass {
        let mut conv = vec![0.0; F * C * C];
        for f in 0..F {
            for i in 0..C {
                for j in 0..C {
                    let mut s = self.bc[f];
                    for a in 0..3 {
                        for b in 0..3 {
                            s += self.k[f * 9 + a * 3 + b] * x[(i + a) * 28 + j + b];
                        }
                    }
                    conv[(f * C + i) * C + j] = s.max(0.0);
                }
            }
        }
        let mut flat = vec![0.0; FLAT];
        let mut arg = vec![0; FLAT];
        for f in 0..F {
            for i in 0..P {
                for j in 0..P {
                    let m = f * P * P + i * P + j;
                    let mut best = (f64::MIN, 0);
                    for (di, dj) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let c = (f * C + 2 * i + di) * C + 2 * j + dj;
                        if conv[c] > best.0 {
                            best = (conv[c], c);
                        }
                    }
                    flat[m] = best.0;
                    arg[m] = best.1;
                }
            }
        }
        let mut z = self.bd.clone();
        for m in 0..FLAT {
            if flat[m] != 0.0 {
                for o in 0..10 {
                    z[o] += flat[m] * self.w[m * 10 + o];
                }
            }
        }
        let mx = z.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
        let s: f64 = e.iter().sum();
        Pass { conv, flat, arg, out: e.iter().map(|v| v / s).collect() }
    }

    fn predict(&self, x: &[f64]) -> usize {
        let o = self.forward(x).out;
        (0..10).fold(0, |b, i| if o[i] > o[b] { i } else { b })
    }

    fn accuracy(&self, xs: &[Vec<f64>], ys: &[usize]) -> f64 {
        xs.iter().zip(ys).filter(|(x, y)| self.predict(x) == **y).count() as f64 / xs.len() as f64
    }

    /// Add one example's gradients into `g` (same layout as the net).
    fn backward(&self, x: &[f64], y: usize, g: &mut Net) {
        let p = self.forward(x);
        let mut dz = p.out.clone();
        dz[y] -= 1.0;
        let mut dconv = vec![0.0; F * C * C];
        for m in 0..FLAT {
            let mut df = 0.0;
            for o in 0..10 {
                g.w[m * 10 + o] += p.flat[m] * dz[o];
                df += self.w[m * 10 + o] * dz[o];
            }
            if p.flat[m] > 0.0 {
                dconv[p.arg[m]] += df;
            }
        }
        for o in 0..10 {
            g.bd[o] += dz[o];
        }
        for f in 0..F {
            for i in 0..C {
                for j in 0..C {
                    let d = dconv[(f * C + i) * C + j];
                    if d != 0.0 && p.conv[(f * C + i) * C + j] > 0.0 {
                        g.bc[f] += d;
                        for a in 0..3 {
                            for b in 0..3 {
                                g.k[f * 9 + a * 3 + b] += d * x[(i + a) * 28 + j + b];
                            }
                        }
                    }
                }
            }
        }
    }

    fn zero(&self) -> Net {
        Net { k: vec![0.0; self.k.len()], bc: vec![0.0; F], w: vec![0.0; self.w.len()], bd: vec![0.0; 10] }
    }

    fn parts(&mut self) -> [&mut Vec<f64>; 4] {
        [&mut self.k, &mut self.bc, &mut self.w, &mut self.bd]
    }

    fn rounded(&self) -> Net {
        let r = |v: &Vec<f64>| v.iter().map(|x| (x * 1e5).round() / 1e5).collect();
        Net { k: r(&self.k), bc: r(&self.bc), w: r(&self.w), bd: r(&self.bd) }
    }
}

fn num(x: f64, places: usize) -> String {
    let s = format!("{x:.places$}");
    let s = s.trim_end_matches('0');
    let s = if s.ends_with('.') { format!("{s}0") } else { s.to_string() };
    if s == "-0.0" { "0.0".into() } else { s }
}

/// `name` bound to the numbers `v`, 40 to a line (joined with c_at),
/// then reshaped to `shape` if given.
fn bind(name: &str, shape: &str, v: &[f64], places: usize) -> String {
    let mut s = String::new();
    for (k, c) in v.chunks(40).enumerate() {
        let c: Vec<String> = c.iter().map(|&x| num(x, places)).collect();
        s += &match k {
            0 => format!("{name} := {}\n", c.join(" ")),
            _ => format!("{name} := {name} c_at {}\n", c.join(" ")),
        };
    }
    if !shape.is_empty() {
        s += &format!("{name} := ({shape}) r_eshape {name}\n");
    }
    s
}

fn main() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../work/mnist");
    let (xs, ys) = mnist(dir, "train");
    let (tx, ty) = mnist(dir, "t10k");
    let mut rng = Rng(0x0dd_b1a5_e5_c0ffee);
    let mut net = Net::new(&mut rng);
    let mut vel = net.zero();
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let (batch, lr, mom) = (32, 0.02, 0.9);
    for epoch in 0..3 {
        for i in (1..order.len()).rev() {
            let j = (rng.next() * (i + 1) as f64) as usize;
            order.swap(i, j);
        }
        for chunk in order.chunks(batch) {
            let mut g = net.zero();
            for &i in chunk {
                net.backward(&xs[i], ys[i], &mut g);
            }
            for (p, (v, g)) in net.parts().into_iter().zip(vel.parts().into_iter().zip(g.parts())) {
                for k in 0..p.len() {
                    v[k] = mom * v[k] - lr * g[k] / chunk.len() as f64;
                    p[k] += v[k];
                }
            }
        }
        eprintln!("pass {}: test accuracy {:.4}", epoch + 1, net.accuracy(&tx, &ty));
    }
    let net = net.rounded();
    let acc = net.accuracy(&tx, &ty);
    eprintln!("written weights (rounded): test accuracy {acc:.4}");
    // One test digit of each class, the first in the test set.
    let samples: Vec<usize> = (0..10).map(|d| ty.iter().position(|&y| y == d).unwrap()).collect();
    let picked: Vec<f64> = samples.iter().flat_map(|&i| tx[i].iter().map(|v| (v * 100.0).round() / 100.0)).collect();
    let section = format!(
        "# -- the weights (written by train/: just cnn-train) ---------\n\
         # Trained on the 60,000 MNIST training digits; {} of the 10,000\n\
         # test digits ({:.2}%) are classified right.\n\
         # k: 8 filters of 3 x 3; bc: their biases; w: 1352 x 10, one row\n\
         # per pooled value (filter by filter, row by row); bd: 10 biases.\n\
         {}{}{}{}\
         # Ten test digits, one of each class (0 to 9), 28 x 28 each.\n{}",
        (acc * 10000.0).round(),
        acc * 100.0,
        bind("k", "8 c_at 3 c_at 3", &net.k, 5),
        bind("bc", "", &net.bc, 5),
        bind("w", "1352 c_at 10", &net.w, 5),
        bind("bd", "", &net.bd, 5),
        bind("samples", "10 c_at 28 c_at 28", &picked, 2),
    );
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../cnn-digits.xtl");
    let src = std::fs::read_to_string(path).expect("read cnn-digits.xtl");
    let (a, b) = (src.find("# -- the weights").expect("no weights marker"), src.find("# -- end of the weights").expect("no end marker"));
    std::fs::write(path, format!("{}{}{}", &src[..a], section, &src[b..])).expect("write cnn-digits.xtl");
    eprintln!("wrote the weights into {path}");
}
