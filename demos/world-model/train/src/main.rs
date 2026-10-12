//! Trains the world-model demo's network and writes its weights into
//! `../data/`, where `../world-model.xtl` reads them.
//!
//! The world: a 2 x 2 ball in a 12 x 12 box, at (x, y) from 0 to 10,
//! moving by (vx, vy), each -2, -1, 1 or 2, bouncing off the walls. Every
//! one of its 11 x 11 x 16 = 1,936 states gives a window: three frames
//! (the state and the next two), and the frame after them to predict.
//! A fifth of the windows, drawn by a fixed seed, are held out: the
//! network never trains on them, so how it does on them says whether it
//! learned the motion or only the windows it saw.
//!
//! The network: 432 pixels in, 64 ReLU units, 144 sigmoid pixels out,
//! trained on the pixels' cross-entropy with Adam, minibatches of 32,
//! a fixed seed. std only. `cargo run --release` (`just world-model-data`).

/// The box's side, the ball's, and the hidden layer's width.
const S: usize = 12;
const BALL: usize = 2;
const H: usize = 64;
const IN: usize = 3 * S * S;
const OUT: usize = S * S;
/// Weight decay (AdamW).
const WD: f64 = 0.2;
const SPEEDS: [i64; 4] = [-2, -1, 1, 2];

/// splitmix64.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// One step of the world on one axis: move, and bounce off a wall.
fn bounce(p: i64, v: i64) -> (i64, i64) {
    let (n, top) = (p + v, (S - BALL) as i64);
    match n {
        n if n < 0 => (-n, -v),
        n if n > top => (2 * top - n, -v),
        n => (n, v),
    }
}

fn frame(x: i64, y: i64) -> Vec<f64> {
    let mut f = vec![0.0; OUT];
    for r in 0..BALL {
        for c in 0..BALL {
            f[(y as usize + r) * S + x as usize + c] = 1.0;
        }
    }
    f
}

/// A state's window: its three frames, and the fourth.
fn window(x: i64, y: i64, vx: i64, vy: i64) -> (Vec<f64>, Vec<f64>) {
    let (mut s, mut frames) = ((x, y, vx, vy), vec![]);
    for _ in 0..4 {
        frames.push(frame(s.0, s.1));
        let ((nx, nvx), (ny, nvy)) = (bounce(s.0, s.2), bounce(s.1, s.3));
        s = (nx, ny, nvx, nvy);
    }
    (frames[..3].concat(), frames[3].clone())
}

struct Net {
    w1: Vec<f64>, // IN x H
    b1: Vec<f64>,
    w2: Vec<f64>, // H x OUT
    b2: Vec<f64>,
}

impl Net {
    fn new(rng: &mut Rng) -> Net {
        let mut init = |n: usize, fan: usize| (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) * (6.0 / fan as f64).sqrt()).collect::<Vec<_>>();
        Net { w1: init(IN * H, IN), b1: vec![0.01; H], w2: init(H * OUT, H + OUT), b2: vec![0.0; OUT] }
    }

    fn hidden(&self, x: &[f64]) -> Vec<f64> {
        let mut h = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                for j in 0..H {
                    h[j] += xi * self.w1[i * H + j];
                }
            }
        }
        h.iter().map(|&v| v.max(0.0)).collect()
    }

    fn predict(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let h = self.hidden(x);
        let mut z = self.b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            for k in 0..OUT {
                z[k] += hj * self.w2[j * OUT + k];
            }
        }
        (h, z.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }

    fn params(&mut self) -> [&mut Vec<f64>; 4] {
        [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }

    /// Weights rounded to what the files keep.
    fn rounded(&self) -> Net {
        let r = |v: &Vec<f64>| v.iter().map(|&w| format!("{w:.6}").parse().unwrap()).collect();
        Net { w1: r(&self.w1), b1: r(&self.b1), w2: r(&self.w2), b2: r(&self.b2) }
    }
}

/// The gradients of the mean cross-entropy over a batch.
fn grads(net: &Net, batch: &[&(Vec<f64>, Vec<f64>)]) -> ([Vec<f64>; 4], f64) {
    let mut g = [vec![0.0; IN * H], vec![0.0; H], vec![0.0; H * OUT], vec![0.0; OUT]];
    let mut loss = 0.0;
    let n = batch.len() as f64;
    for (x, y) in batch {
        let (h, p) = net.predict(x);
        let dz: Vec<f64> = p.iter().zip(y).map(|(p, y)| (p - y) / n).collect();
        loss -= p.iter().zip(y).map(|(p, y)| y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln()).sum::<f64>() / n;
        let mut dh = vec![0.0; H];
        for j in 0..H {
            for k in 0..OUT {
                g[2][j * OUT + k] += h[j] * dz[k];
                dh[j] += net.w2[j * OUT + k] * dz[k];
            }
        }
        for k in 0..OUT {
            g[3][k] += dz[k];
        }
        for j in 0..H {
            if h[j] <= 0.0 {
                dh[j] = 0.0;
            }
            g[1][j] += dh[j];
        }
        for (i, &xi) in x.iter().enumerate() {
            if xi != 0.0 {
                for j in 0..H {
                    g[0][i * H + j] += xi * dh[j];
                }
            }
        }
    }
    (g, loss)
}

/// Predicted exactly: every pixel on the right side of one half.
fn exact(net: &Net, set: &[&(Vec<f64>, Vec<f64>)]) -> usize {
    set.iter().filter(|(x, y)| net.predict(x).1.iter().zip(y.iter()).all(|(p, y)| (*p > 0.5) == (*y > 0.5))).count()
}

fn write(dir: &str, name: &str, v: &[f64], cols: usize) {
    let text: String = v.chunks(cols).map(|row| row.iter().map(|w| format!("{w:.6}")).collect::<Vec<_>>().join(" ") + "\n").collect();
    std::fs::write(format!("{dir}/{name}.txt"), text).unwrap_or_else(|e| panic!("{dir}/{name}.txt: {e}"));
}

fn main() {
    let t0 = std::time::Instant::now();
    let data = format!("{}/../data", env!("CARGO_MANIFEST_DIR"));
    let mut rng = Rng(20261012);
    let mut all = vec![];
    for x in 0..=(S - BALL) as i64 {
        for y in 0..=(S - BALL) as i64 {
            for vx in SPEEDS {
                for vy in SPEEDS {
                    all.push((window(x, y, vx, vy), (x, y, vx, vy)));
                }
            }
        }
    }
    // A fifth held out, by a shuffle.
    for i in (1..all.len()).rev() {
        let j = rng.below(i + 1);
        all.swap(i, j);
    }
    let held = all.len() / 5;
    let test: Vec<&(Vec<f64>, Vec<f64>)> = all[..held].iter().map(|(w, _)| w).collect();
    let mut train: Vec<&(Vec<f64>, Vec<f64>)> = all[held..].iter().map(|(w, _)| w).collect();
    let mut net = Net::new(&mut rng);
    let (mut m, mut v): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (net.params().iter().map(|p| vec![0.0; p.len()]).collect(), net.params().iter().map(|p| vec![0.0; p.len()]).collect());
    let (epochs, batch) = (400, 32);
    let mut t = 0;
    for epoch in 1..=epochs {
        for i in (1..train.len()).rev() {
            let j = rng.below(i + 1);
            train.swap(i, j);
        }
        let lr = 0.002 * (1.0 - 0.9 * (epoch - 1) as f64 / epochs as f64);
        let mut total = 0.0;
        for b in train.chunks(batch) {
            let (g, loss) = grads(&net, b);
            total += loss * b.len() as f64;
            t += 1;
            for (k, p) in net.params().into_iter().enumerate() {
                for i in 0..p.len() {
                    m[k][i] = 0.9 * m[k][i] + 0.1 * g[k][i];
                    v[k][i] = 0.999 * v[k][i] + 0.001 * g[k][i] * g[k][i];
                    let (mh, vh) = (m[k][i] / (1.0 - 0.9f64.powi(t)), v[k][i] / (1.0 - 0.999f64.powi(t)));
                    p[i] -= lr * (mh / (vh.sqrt() + 1e-8) + WD * p[i]);
                }
            }
        }
        if epoch % 50 == 0 {
            println!("epoch {epoch}: loss {:.5}, exact {}/{} trained, {}/{} held out ({:.0?})", total / train.len() as f64, exact(&net, &train), train.len(), exact(&net, &test), test.len(), t0.elapsed());
        }
    }
    let r = net.rounded();
    println!("as written (6 decimals): exact {}/{} trained, {}/{} held out", exact(&r, &train), train.len(), exact(&r, &test), test.len());
    write(&data, "w1", &r.w1, H);
    write(&data, "b1", &r.b1, H);
    write(&data, "w2", &r.w2, OUT);
    write(&data, "b2", &r.b2, OUT);
    // The held-out states (x y vx vy, one per line), which the program checks.
    let text: String = all[..held].iter().map(|(_, (x, y, vx, vy))| format!("{x} {y} {vx} {vy}\n")).collect();
    std::fs::write(format!("{data}/held-out.txt"), text).expect("held-out.txt");
    // One window's predicted pixels, which X_eTaL's forward pass must match.
    let (x, _) = window(3, 7, 2, -1);
    write(&data, "expected", &r.predict(&x).1, S);
    println!("wrote {data} in {:.0?}", t0.elapsed());
}
