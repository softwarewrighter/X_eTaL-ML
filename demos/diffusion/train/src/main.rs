//! Trains the diffusion demo's denoiser and writes its weights into
//! `../data/`, where `../diffusion.xtl` reads them.
//!
//! The pictures: 10 x 10, pixels -1 (off) or 1 (on), one of three shapes
//! at a random place: a filled disk, a hollow square, a cross; made
//! fresh for every batch (there is no end of them). The noise schedule:
//! T = 10 levels, the kept share of the picture ab(t) = cos(pi/2 t/T)^2
//! (at least 0.001), the noisy picture sqrt(ab) x + sqrt(1 - ab) e for
//! a Gaussian e. The denoiser: the noisy picture and the level as a
//! one-hot (110 in), 256 ReLU units, 100 sigmoid outputs, the clean
//! picture's probability per pixel, trained on cross-entropy with Adam.
//!
//! Sampling is DDIM with no added noise: from a Gaussian picture at
//! T, each step predicts the clean picture, infers the noise, and
//! steps to the next level, so the only chance is the start. The
//! demo's program and page do the same in X_eTaL; this trainer writes
//! one starting picture and what it must become (`noise.txt`,
//! `expected.txt`). std only, a fixed seed. `cargo run --release`
//! (`just diffusion-data`).

const S: usize = 10;
const N: usize = S * S;
const T: usize = 10;
const IN: usize = N + T;
const H: usize = 256;

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
        ((self.next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    /// Box-Muller, as the X_eTaL program draws its noise.
    fn gaussian(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// The kept share of the picture at level t (0 is the clean picture).
fn ab(t: usize) -> f64 {
    (std::f64::consts::FRAC_PI_2 * t as f64 / T as f64).cos().powi(2).max(0.001)
}

/// A picture of one of the three shapes.
fn picture(rng: &mut Rng) -> Vec<f64> {
    let mut p = vec![-1.0; N];
    let on = |p: &mut Vec<f64>, r: i64, c: i64| {
        if (0..S as i64).contains(&r) && (0..S as i64).contains(&c) {
            p[r as usize * S + c as usize] = 1.0;
        }
    };
    match rng.below(3) {
        0 => {
            let (cr, cc, rad) = (2.5 + rng.uniform() * 4.0, 2.5 + rng.uniform() * 4.0, 1.5 + rng.uniform() * 1.5);
            for r in 0..S {
                for c in 0..S {
                    if (r as f64 + 0.5 - cr).powi(2) + (c as f64 + 0.5 - cc).powi(2) <= rad * rad {
                        p[r * S + c] = 1.0;
                    }
                }
            }
        }
        1 => {
            let side = 4 + rng.below(3) as i64;
            let (r0, c0) = (rng.below(S + 1 - side as usize) as i64, rng.below(S + 1 - side as usize) as i64);
            for i in 0..side {
                on(&mut p, r0, c0 + i);
                on(&mut p, r0 + side - 1, c0 + i);
                on(&mut p, r0 + i, c0);
                on(&mut p, r0 + i, c0 + side - 1);
            }
        }
        _ => {
            let arm = 2 + rng.below(2) as i64;
            let (cr, cc) = ((arm + rng.below(S - 2 * arm as usize) as i64), (arm + rng.below(S - 2 * arm as usize) as i64));
            for i in -arm..=arm {
                on(&mut p, cr, cc + i);
                on(&mut p, cr + i, cc);
            }
        }
    }
    p
}

struct Net {
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
}

impl Net {
    fn new(rng: &mut Rng) -> Net {
        let mut init = |n: usize, fan: usize| (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) * (6.0 / fan as f64).sqrt()).collect::<Vec<_>>();
        Net { w1: init(IN * H, IN), b1: vec![0.01; H], w2: init(H * N, H + N), b2: vec![0.0; N] }
    }

    /// The hidden layer and the output probabilities for the noisy picture x at level t.
    fn predict(&self, x: &[f64], t: usize) -> (Vec<f64>, Vec<f64>) {
        let mut h = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            for j in 0..H {
                h[j] += xi * self.w1[i * H + j];
            }
        }
        for j in 0..H {
            h[j] += self.w1[(N + t - 1) * H + j];
        }
        let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
        let mut z = self.b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            if hj != 0.0 {
                for k in 0..N {
                    z[k] += hj * self.w2[j * N + k];
                }
            }
        }
        (h, z.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }

    fn params(&mut self) -> [&mut Vec<f64>; 4] {
        [&mut self.w1, &mut self.b1, &mut self.w2, &mut self.b2]
    }

    fn rounded(&self) -> Net {
        let r = |v: &Vec<f64>| v.iter().map(|&w| format!("{w:.6}").parse().unwrap()).collect();
        Net { w1: r(&self.w1), b1: r(&self.b1), w2: r(&self.w2), b2: r(&self.b2) }
    }

    /// DDIM from the Gaussian picture e: the clean picture's probabilities.
    fn sample(&self, e: &[f64]) -> Vec<f64> {
        let mut x = e.to_vec();
        let mut x0 = vec![0.0; N];
        for t in (1..=T).rev() {
            let p = self.predict(&x, t).1;
            x0 = p.iter().map(|&p| 2.0 * p - 1.0).collect();
            let (a, b) = (ab(t), ab(t - 1));
            x = (0..N).map(|k| b.sqrt() * x0[k] + (1.0 - b).sqrt() * (x[k] - a.sqrt() * x0[k]) / (1.0 - a).sqrt()).collect();
        }
        x0
    }
}

fn show(p: &[f64]) -> String {
    p.chunks(S).map(|r| r.iter().map(|&v| if v > 0.0 { '#' } else { '.' }).collect::<String>() + "\n").collect()
}

/// `digits` decimals: 6 for the weights (what the files keep; sampling uses them rounded), 15 for a result.
fn write(dir: &str, name: &str, v: &[f64], cols: usize, digits: usize) {
    let text: String = v.chunks(cols).map(|row| row.iter().map(|w| format!("{w:.*}", digits)).collect::<Vec<_>>().join(" ") + "\n").collect();
    std::fs::write(format!("{dir}/{name}.txt"), text).unwrap_or_else(|e| panic!("{dir}/{name}.txt: {e}"));
}

fn main() {
    let t0 = std::time::Instant::now();
    let data = format!("{}/../data", env!("CARGO_MANIFEST_DIR"));
    let mut rng = Rng(20261013);
    let mut net = Net::new(&mut rng);
    let (mut m, mut v): (Vec<Vec<f64>>, Vec<Vec<f64>>) = (net.params().iter().map(|p| vec![0.0; p.len()]).collect(), net.params().iter().map(|p| vec![0.0; p.len()]).collect());
    let (steps, batch) = (40000, 64);
    let mut recent = 0.0;
    for step in 1..=steps {
        let mut g = [vec![0.0; IN * H], vec![0.0; H], vec![0.0; H * N], vec![0.0; N]];
        let mut loss = 0.0;
        for _ in 0..batch {
            let x0 = picture(&mut rng);
            let t = 1 + rng.below(T);
            let a = ab(t);
            let x: Vec<f64> = x0.iter().map(|&v| a.sqrt() * v + (1.0 - a).sqrt() * rng.gaussian()).collect();
            let (h, p) = net.predict(&x, t);
            let y: Vec<f64> = x0.iter().map(|&v| (v + 1.0) / 2.0).collect();
            loss -= p.iter().zip(&y).map(|(p, y)| y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln()).sum::<f64>() / batch as f64;
            let dz: Vec<f64> = p.iter().zip(&y).map(|(p, y)| (p - y) / batch as f64).collect();
            let mut dh = vec![0.0; H];
            for j in 0..H {
                if h[j] > 0.0 {
                    for k in 0..N {
                        g[2][j * N + k] += h[j] * dz[k];
                        dh[j] += net.w2[j * N + k] * dz[k];
                    }
                }
            }
            for k in 0..N {
                g[3][k] += dz[k];
            }
            for j in 0..H {
                if dh[j] != 0.0 {
                    g[1][j] += dh[j];
                    for (i, &xi) in x.iter().enumerate() {
                        g[0][i * H + j] += xi * dh[j];
                    }
                    g[0][(N + t - 1) * H + j] += dh[j];
                }
            }
        }
        let lr = 0.002 * (1.0 - 0.9 * (step - 1) as f64 / steps as f64);
        for (k, p) in net.params().into_iter().enumerate() {
            for i in 0..p.len() {
                m[k][i] = 0.9 * m[k][i] + 0.1 * g[k][i];
                v[k][i] = 0.999 * v[k][i] + 0.001 * g[k][i] * g[k][i];
                let (mh, vh) = (m[k][i] / (1.0 - 0.9f64.powi(step as i32)), v[k][i] / (1.0 - 0.999f64.powi(step as i32)));
                p[i] -= lr * mh / (vh.sqrt() + 1e-8);
            }
        }
        recent = 0.98 * recent + 0.02 * loss;
        if step % 10000 == 0 {
            println!("step {step}: loss {recent:.4} ({:.0?})", t0.elapsed());
        }
    }
    let r = net.rounded();
    for _ in 0..6 {
        let e: Vec<f64> = (0..N).map(|_| rng.gaussian()).collect();
        println!("{}", show(&r.sample(&e)));
    }
    write(&data, "w1", &r.w1, H, 6);
    write(&data, "b1", &r.b1, H, 6);
    write(&data, "w2", &r.w2, N, 6);
    write(&data, "b2", &r.b2, N, 6);
    let e: Vec<f64> = (0..N).map(|_| rng.gaussian()).collect();
    write(&data, "noise", &e, S, 15);
    write(&data, "expected", &r.sample(&e), S, 15);
    println!("wrote {data} in {:.0?}", t0.elapsed());
}
