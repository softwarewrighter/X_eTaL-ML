//! Trains the diffusion demo's denoiser and writes its weights into
//! `../data/`, where `../diffusion.xtl` reads them.
//!
//! The pictures: 10 x 10, pixels -1 (off) or 1 (on), one of three shapes
//! at a random place: a filled disk, a hollow square, a cross; made
//! fresh for every batch (there is no end of them). The noise schedule:
//! T = 10 levels, the kept share of the picture ab(t) = cos(pi/2 t/T)^2
//! (at least 0.001), the noisy picture sqrt(ab) x + sqrt(1 - ab) e for
//! a Gaussian e. The denoiser: a 3 x 3 convolution of the noisy picture
//! (8 filters, valid, so 8 maps of 8 x 8) with ReLU, its 512 features
//! through a dense layer of 256 ReLU units to which the level adds its
//! own weights (a row per level), and 100 sigmoid outputs, the clean
//! picture's probability per pixel, trained on cross-entropy with Adam.
//! `cargo run --release -- dense` trains the earlier denoiser instead
//! (the 100 pixels straight into the dense layer) and only reports, so
//! the two can be compared on the same held-out pictures; `pool` pools
//! the maps by 2 x 2 blocks first (an experiment that lost: 13.64
//! against 13.36 dense and 12.77 convolutional); a number is the steps
//! (100,000).
//!
//! Sampling is DDIM with no added noise: from a Gaussian picture at
//! T, each step predicts the clean picture, infers the noise, and
//! steps to the next level, so the only chance is the start. The
//! demo's program and page do the same in X_eTaL; this trainer writes
//! one starting picture and what it must become (`noise.txt`,
//! `expected.txt`). std only, a fixed seed; the batch's gradients are
//! summed over eight threads. `cargo run --release` (`just diffusion-data`).

const S: usize = 10;
const N: usize = S * S;
const T: usize = 10;
/// The filters, and each map's side (valid: no padding).
const F: usize = 8;
const R: usize = S - 2;
const FEAT: usize = F * R * R;
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

/// A noisy picture at a level, and the clean picture's pixels as 0 or 1.
struct Sample {
    x: Vec<f64>,
    t: usize,
    y: Vec<f64>,
}

fn sample(rng: &mut Rng) -> Sample {
    let x0 = picture(rng);
    let t = 1 + rng.below(T);
    let a = ab(t);
    let x = x0.iter().map(|&v| a.sqrt() * v + (1.0 - a).sqrt() * rng.gaussian()).collect();
    Sample { x, t, y: x0.iter().map(|&v| (v + 1.0) / 2.0).collect() }
}

/// The parameters, in one order: the filters, their biases, the first
/// dense layer's weights, the level's rows, its biases, the output
/// layer's weights, its biases. The dense denoiser has no filters (its
/// first layer takes the pixels).
#[derive(Clone)]
struct Net {
    conv: bool,
    /// The maps pooled by 2 x 2 blocks before the dense layer (an experiment: `pool`).
    pool: bool,
    p: [Vec<f64>; 7],
}

/// The pooled maps' side and count.
const RP: usize = R / 2;
const POOLED: usize = F * RP * RP;

const K: usize = 0;
const BC: usize = 1;
const W1: usize = 2;
const WH: usize = 3;
const B1: usize = 4;
const W2: usize = 5;
const B2: usize = 6;

impl Net {
    fn new(rng: &mut Rng, conv: bool, pool: bool) -> Net {
        let mut init = |n: usize, fan: usize| (0..n).map(|_| (rng.uniform() * 2.0 - 1.0) * (6.0 / fan as f64).sqrt()).collect::<Vec<_>>();
        let inputs = if !conv { N } else if pool { POOLED } else { FEAT };
        let (k, bc) = if conv { (init(F * 9, 9), vec![0.0; F]) } else { (vec![], vec![]) };
        Net { conv, pool, p: [k, bc, init(inputs * H, inputs), init(T * H, inputs), vec![0.01; H], init(H * N, H + N), vec![0.0; N]] }
    }

    /// The largest of each 2 x 2 block of the maps, and where it is.
    fn pooled(maps: &[f64]) -> (Vec<f64>, Vec<usize>) {
        let (mut v, mut at) = (vec![0.0; POOLED], vec![0; POOLED]);
        for f in 0..F {
            for r in 0..RP {
                for c in 0..RP {
                    let i = (f * RP + r) * RP + c;
                    at[i] = (f * R + 2 * r) * R + 2 * c;
                    for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                        let j = (f * R + 2 * r + dr) * R + 2 * c + dc;
                        if maps[j] > maps[at[i]] {
                            at[i] = j;
                        }
                    }
                    v[i] = maps[at[i]];
                }
            }
        }
        (v, at)
    }

    fn zero(&self) -> [Vec<f64>; 7] {
        self.p.each_ref().map(|v| vec![0.0; v.len()])
    }

    /// The first layer's inputs for the noisy picture x: the convolution's
    /// maps after ReLU (map by map, row by row), or the pixels themselves.
    fn features(&self, x: &[f64]) -> Vec<f64> {
        if !self.conv {
            return x.to_vec();
        }
        let v = self.maps(x);
        if self.pool {
            return Net::pooled(&v).0;
        }
        v
    }

    /// The convolution's maps after ReLU (map by map, row by row).
    fn maps(&self, x: &[f64]) -> Vec<f64> {
        let (k, bc) = (&self.p[K], &self.p[BC]);
        let mut v = vec![0.0; FEAT];
        for f in 0..F {
            for r in 0..R {
                for c in 0..R {
                    let mut s = bc[f];
                    for dr in 0..3 {
                        for dc in 0..3 {
                            s += k[f * 9 + dr * 3 + dc] * x[(r + dr) * S + c + dc];
                        }
                    }
                    v[(f * R + r) * R + c] = s.max(0.0);
                }
            }
        }
        v
    }

    /// The features, the hidden layer and the output probabilities for the noisy picture x at level t.
    fn predict(&self, x: &[f64], t: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let v = self.features(x);
        let (w1, wh, b1, w2, b2) = (&self.p[W1], &self.p[WH], &self.p[B1], &self.p[W2], &self.p[B2]);
        let mut h = b1.clone();
        for (i, &vi) in v.iter().enumerate() {
            if vi != 0.0 {
                for j in 0..H {
                    h[j] += vi * w1[i * H + j];
                }
            }
        }
        for j in 0..H {
            h[j] += wh[(t - 1) * H + j];
        }
        let h: Vec<f64> = h.iter().map(|&v| v.max(0.0)).collect();
        let mut z = b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            if hj != 0.0 {
                for k in 0..N {
                    z[k] += hj * w2[j * N + k];
                }
            }
        }
        (v, h, z.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }

    /// The cross-entropy of one sample, and its gradients (scaled) added into g.
    fn backward(&self, s: &Sample, scale: f64, g: &mut [Vec<f64>; 7]) -> f64 {
        let (v, h, p) = self.predict(&s.x, s.t);
        let loss = -p.iter().zip(&s.y).map(|(p, y)| y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln()).sum::<f64>();
        let dz: Vec<f64> = p.iter().zip(&s.y).map(|(p, y)| (p - y) * scale).collect();
        let (w1, w2) = (&self.p[W1], &self.p[W2]);
        let mut dh = vec![0.0; H];
        for j in 0..H {
            if h[j] > 0.0 {
                for k in 0..N {
                    g[W2][j * N + k] += h[j] * dz[k];
                    dh[j] += w2[j * N + k] * dz[k];
                }
            }
        }
        for k in 0..N {
            g[B2][k] += dz[k];
        }
        let mut dv = vec![0.0; v.len()];
        for j in 0..H {
            if dh[j] != 0.0 {
                g[B1][j] += dh[j];
                g[WH][(s.t - 1) * H + j] += dh[j];
                for (i, &vi) in v.iter().enumerate() {
                    if vi != 0.0 {
                        g[W1][i * H + j] += vi * dh[j];
                        dv[i] += w1[i * H + j] * dh[j];
                    }
                }
            }
        }
        if self.conv {
            // Pooled: each block's gradient goes to its largest item.
            let dm = if self.pool {
                let (_, at) = Net::pooled(&self.maps(&s.x));
                let mut dm = vec![0.0; FEAT];
                for (i, &j) in at.iter().enumerate() {
                    dm[j] += dv[i];
                }
                dm
            } else {
                dv
            };
            for f in 0..F {
                for r in 0..R {
                    for c in 0..R {
                        let d = dm[(f * R + r) * R + c];
                        if d != 0.0 {
                            g[BC][f] += d;
                            for dr in 0..3 {
                                for dc in 0..3 {
                                    g[K][f * 9 + dr * 3 + dc] += s.x[(r + dr) * S + c + dc] * d;
                                }
                            }
                        }
                    }
                }
            }
        }
        loss
    }

    fn rounded(&self) -> Net {
        Net { conv: self.conv, pool: self.pool, p: self.p.each_ref().map(|v| v.iter().map(|&w| format!("{w:.6}").parse().unwrap()).collect()) }
    }

    /// DDIM from the Gaussian picture e: the clean picture's probabilities.
    fn sample(&self, e: &[f64]) -> Vec<f64> {
        let mut x = e.to_vec();
        let mut x0 = vec![0.0; N];
        for t in (1..=T).rev() {
            let p = self.predict(&x, t).2;
            x0 = p.iter().map(|&p| 2.0 * p - 1.0).collect();
            let (a, b) = (ab(t), ab(t - 1));
            x = (0..N).map(|k| b.sqrt() * x0[k] + (1.0 - b).sqrt() * (x[k] - a.sqrt() * x0[k]) / (1.0 - a).sqrt()).collect();
        }
        x0
    }

    /// The mean cross-entropy per picture over 2,000 held-out pictures
    /// (the same ones for every net: their own seed), and per level.
    fn held_out(&self) -> (f64, Vec<f64>) {
        let mut rng = Rng(20261014);
        let (mut total, mut per) = (0.0, vec![(0.0, 0usize); T]);
        for _ in 0..2000 {
            let s = sample(&mut rng);
            let p = self.predict(&s.x, s.t).2;
            let loss = -p.iter().zip(&s.y).map(|(p, y)| y * p.max(1e-12).ln() + (1.0 - y) * (1.0 - p).max(1e-12).ln()).sum::<f64>();
            total += loss;
            per[s.t - 1].0 += loss;
            per[s.t - 1].1 += 1;
        }
        (total / 2000.0, per.iter().map(|(l, n)| l / *n as f64).collect())
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
    // `dense` trains the earlier denoiser (reported, not written); a number is the steps (40,000).
    let args: Vec<String> = std::env::args().skip(1).collect();
    let conv = !args.iter().any(|a| a == "dense");
    let pool = args.iter().any(|a| a == "pool");
    let steps: usize = args.iter().find_map(|a| a.parse().ok()).unwrap_or(100000);
    let data = format!("{}/../data", env!("CARGO_MANIFEST_DIR"));
    let mut rng = Rng(20261013);
    let mut net = Net::new(&mut rng, conv, pool);
    let (mut m, mut v) = (net.zero(), net.zero());
    let (batch, threads) = (64, 8);
    let mut recent = 0.0;
    for step in 1..=steps {
        // The batch is drawn in order (the seed decides it); its gradients are summed by chunk, in order.
        let samples: Vec<Sample> = (0..batch).map(|_| sample(&mut rng)).collect();
        let scale = 1.0 / batch as f64;
        let parts: Vec<([Vec<f64>; 7], f64)> = std::thread::scope(|sc| {
            let net = &net;
            let jobs: Vec<_> = samples
                .chunks(batch / threads)
                .map(|chunk| {
                    sc.spawn(move || {
                        let mut g = net.zero();
                        let loss = chunk.iter().map(|s| net.backward(s, scale, &mut g)).sum::<f64>();
                        (g, loss)
                    })
                })
                .collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let mut g = net.zero();
        let mut loss = 0.0;
        for (pg, pl) in parts {
            for (k, v) in pg.into_iter().enumerate() {
                for (a, b) in g[k].iter_mut().zip(v) {
                    *a += b;
                }
            }
            loss += pl * scale;
        }
        let lr = 0.002 * (1.0 - 0.9 * (step - 1) as f64 / steps as f64);
        for k in 0..7 {
            for i in 0..net.p[k].len() {
                m[k][i] = 0.9 * m[k][i] + 0.1 * g[k][i];
                v[k][i] = 0.999 * v[k][i] + 0.001 * g[k][i] * g[k][i];
                let (mh, vh) = (m[k][i] / (1.0 - 0.9f64.powi(step as i32)), v[k][i] / (1.0 - 0.999f64.powi(step as i32)));
                net.p[k][i] -= lr * mh / (vh.sqrt() + 1e-8);
            }
        }
        recent = 0.98 * recent + 0.02 * loss;
        if step % 10000 == 0 {
            println!("step {step}: loss {recent:.4} ({:.0?})", t0.elapsed());
        }
    }
    let r = net.rounded();
    let (held, per) = r.held_out();
    println!(
        "{} denoiser, rounded: held-out cross-entropy {held:.3} per picture; by level 1 to {T}: {}",
        if !conv { "dense" } else if pool { "pooled convolutional" } else { "convolutional" },
        per.iter().map(|l| format!("{l:.2}")).collect::<Vec<_>>().join(" ")
    );
    for _ in 0..6 {
        let e: Vec<f64> = (0..N).map(|_| rng.gaussian()).collect();
        println!("{}", show(&r.sample(&e)));
    }
    if !conv {
        println!("(the dense denoiser is only reported; nothing written)");
        return;
    }
    // The filters as Conv reads them: a row per filter, its 9 weights then its bias.
    let filters: Vec<f64> = (0..F).flat_map(|f| r.p[K][9 * f..9 * f + 9].iter().copied().chain([r.p[BC][f]])).collect();
    write(&data, "filters", &filters, 10, 6);
    write(&data, "w1", &r.p[W1], H, 6);
    write(&data, "wh", &r.p[WH], H, 6);
    write(&data, "b1", &r.p[B1], H, 6);
    write(&data, "w2", &r.p[W2], N, 6);
    write(&data, "b2", &r.p[B2], N, 6);
    let e: Vec<f64> = (0..N).map(|_| rng.gaussian()).collect();
    write(&data, "noise", &e, S, 15);
    write(&data, "expected", &r.sample(&e), S, 15);
    println!("wrote {data} in {:.0?}", t0.elapsed());
}
