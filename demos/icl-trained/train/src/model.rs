//! A small Llama-style transformer (pre-RMSNorm, causal multi-head
//! attention, a SwiGLU feed-forward layer, untied unembedding) with its
//! backward pass written out, one prompt at a time, in f32. One departure
//! from Llama: learned position embeddings in place of rotary ones, so
//! the X_eTaL forward pass stays a few lines.

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub maxlen: usize,
}

pub const EPS: f32 = 1e-6;

/// Where each weight starts in the flat vector.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub g1: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub g2: usize,
    pub wg: usize,
    pub wu: usize,
    pub wd: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub c: Config,
    pub emb: usize,
    pub pos: usize,
    pub layers: Vec<Layer>,
    pub gf: usize,
    pub un: usize,
    pub size: usize,
}

impl Layout {
    pub fn new(c: Config) -> Layout {
        let mut at = 0;
        let mut take = |n: usize| {
            let a = at;
            at += n;
            a
        };
        let (d, f) = (c.d, c.ffn);
        let emb = take(c.vocab * d);
        let pos = take(c.maxlen * d);
        let layers = (0..c.layers)
            .map(|_| Layer { g1: take(d), wq: take(d * d), wk: take(d * d), wv: take(d * d), wo: take(d * d), g2: take(d), wg: take(d * f), wu: take(d * f), wd: take(f * d) })
            .collect();
        let gf = take(d);
        let un = take(d * c.vocab);
        Layout { c, emb, pos, layers, gf, un, size: at }
    }

    /// The norm gains, which start at 1 (the rest at N(0, 0.02)).
    pub fn gains(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.layers.iter().flat_map(|l| [l.g1, l.g2]).collect();
        v.push(self.gf);
        v
    }
}

/// y (n x o) = x (n x i) W (i x o), W row-major at w.
pub fn mm(x: &[f32], n: usize, i: usize, w: &[f32], o: usize) -> Vec<f32> {
    let mut y = vec![0.0; n * o];
    for r in 0..n {
        let yr = &mut y[r * o..(r + 1) * o];
        for k in 0..i {
            let a = x[r * i + k];
            if a == 0.0 {
                continue;
            }
            let wr = &w[k * o..(k + 1) * o];
            for (yv, wv) in yr.iter_mut().zip(wr) {
                *yv += a * wv;
            }
        }
    }
    y
}

/// dW += x^T dy; dx = dy W^T.
fn mm_back(x: &[f32], n: usize, i: usize, w: &[f32], o: usize, dy: &[f32], dw: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0; n * i];
    for r in 0..n {
        let dyr = &dy[r * o..(r + 1) * o];
        for k in 0..i {
            let a = x[r * i + k];
            let dwr = &mut dw[k * o..(k + 1) * o];
            let wr = &w[k * o..(k + 1) * o];
            let mut s = 0.0;
            for j in 0..o {
                dwr[j] += a * dyr[j];
                s += dyr[j] * wr[j];
            }
            dx[r * i + k] = s;
        }
    }
    dx
}

fn rms(x: &[f32], n: usize, d: usize, g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0; n * d];
    let mut inv = vec![0.0; n];
    for r in 0..n {
        let row = &x[r * d..(r + 1) * d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        inv[r] = 1.0 / (ms + EPS).sqrt();
        for j in 0..d {
            y[r * d + j] = row[j] * inv[r] * g[j];
        }
    }
    (y, inv)
}

fn rms_back(x: &[f32], n: usize, d: usize, g: &[f32], inv: &[f32], dy: &[f32], dg: &mut [f32]) -> Vec<f32> {
    let mut dx = vec![0.0; n * d];
    for r in 0..n {
        let s = inv[r];
        let mut dot = 0.0;
        for j in 0..d {
            let xh = x[r * d + j] * s;
            dg[j] += dy[r * d + j] * xh;
            dot += dy[r * d + j] * g[j] * xh;
        }
        dot /= d as f32;
        for j in 0..d {
            let xh = x[r * d + j] * s;
            dx[r * d + j] = s * (dy[r * d + j] * g[j] - xh * dot);
        }
    }
    dx
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// What one layer kept for the backward pass.
struct Saved {
    x: Vec<f32>,
    h: Vec<f32>,
    inv1: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    x2: Vec<f32>,
    h2: Vec<f32>,
    inv2: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    m: Vec<f32>,
}

/// A forward pass's results: the logits at every position, and what the
/// backward pass needs.
pub struct Forward {
    pub logits: Vec<f32>,
    saved: Vec<Saved>,
    xf: Vec<f32>,
    hf: Vec<f32>,
    invf: Vec<f32>,
}

pub fn forward(lay: &Layout, p: &[f32], seq: &[usize]) -> Forward {
    let c = lay.c;
    let (n, d, f, hh) = (seq.len(), c.d, c.ffn, c.heads);
    let dh = d / hh;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut x = vec![0.0; n * d];
    for (t, &tok) in seq.iter().enumerate() {
        for j in 0..d {
            x[t * d + j] = p[lay.emb + tok * d + j] + p[lay.pos + t * d + j];
        }
    }
    let mut saved = vec![];
    for l in &lay.layers {
        let (h, inv1) = rms(&x, n, d, &p[l.g1..l.g1 + d]);
        let q = mm(&h, n, d, &p[l.wq..], d);
        let k = mm(&h, n, d, &p[l.wk..], d);
        let v = mm(&h, n, d, &p[l.wv..], d);
        let mut att = vec![0.0; hh * n * n];
        let mut o = vec![0.0; n * d];
        for hd in 0..hh {
            for t in 0..n {
                let row = &mut att[(hd * n + t) * n..(hd * n + t + 1) * n];
                let mut mx = f32::MIN;
                for s in 0..=t {
                    let mut dot = 0.0;
                    for j in 0..dh {
                        dot += q[t * d + hd * dh + j] * k[s * d + hd * dh + j];
                    }
                    row[s] = dot * scale;
                    mx = mx.max(row[s]);
                }
                let mut z = 0.0;
                for s in 0..=t {
                    row[s] = (row[s] - mx).exp();
                    z += row[s];
                }
                for s in 0..=t {
                    row[s] /= z;
                    for j in 0..dh {
                        o[t * d + hd * dh + j] += row[s] * v[s * d + hd * dh + j];
                    }
                }
            }
        }
        let ao = mm(&o, n, d, &p[l.wo..], d);
        let x2: Vec<f32> = x.iter().zip(&ao).map(|(a, b)| a + b).collect();
        let (h2, inv2) = rms(&x2, n, d, &p[l.g2..l.g2 + d]);
        let a = mm(&h2, n, d, &p[l.wg..], f);
        let b = mm(&h2, n, d, &p[l.wu..], f);
        let m: Vec<f32> = a.iter().zip(&b).map(|(a, b)| a * sigmoid(*a) * b).collect();
        let fo = mm(&m, n, f, &p[l.wd..], d);
        let x3: Vec<f32> = x2.iter().zip(&fo).map(|(a, b)| a + b).collect();
        saved.push(Saved { x, h, inv1, q, k, v, att, o, x2, h2, inv2, a, b, m });
        x = x3;
    }
    let (hf, invf) = rms(&x, n, d, &p[lay.gf..lay.gf + d]);
    let logits = mm(&hf, n, d, &p[lay.un..], c.vocab);
    Forward { logits, saved, xf: x, hf, invf }
}

/// The cross-entropy at the targets (position, token), summed, and its
/// gradient (times `weight`) added into `g`.
pub fn backward(lay: &Layout, p: &[f32], seq: &[usize], fw: &Forward, targets: &[(usize, usize)], weight: f32, g: &mut [f32]) -> f32 {
    let c = lay.c;
    let (n, d, f, hh, vv) = (seq.len(), c.d, c.ffn, c.heads, c.vocab);
    let dh = d / hh;
    let scale = 1.0 / (dh as f32).sqrt();
    let mut dlog = vec![0.0; n * vv];
    let mut loss = 0.0;
    for &(t, y) in targets {
        let row = &fw.logits[t * vv..(t + 1) * vv];
        let mx = row.iter().cloned().fold(f32::MIN, f32::max);
        let z: f32 = row.iter().map(|v| (v - mx).exp()).sum();
        loss += z.ln() + mx - row[y];
        for j in 0..vv {
            dlog[t * vv + j] += weight * (row[j] - mx).exp() / z;
        }
        dlog[t * vv + y] -= weight;
    }
    let dhf = mm_back(&fw.hf, n, d, &p[lay.un..lay.un + d * vv], vv, &dlog, &mut g[lay.un..lay.un + d * vv]);
    let mut dx = {
        let (gf, rest) = g[lay.gf..].split_at_mut(d);
        let _ = rest;
        rms_back(&fw.xf, n, d, &p[lay.gf..lay.gf + d], &fw.invf, &dhf, gf)
    };
    for (l, s) in lay.layers.iter().zip(&fw.saved).rev() {
        // The feed-forward layer.
        let dm = mm_back(&s.m, n, f, &p[l.wd..l.wd + f * d], d, &dx, &mut g[l.wd..l.wd + f * d]);
        let mut da = vec![0.0; n * f];
        let mut db = vec![0.0; n * f];
        for i in 0..n * f {
            let sg = sigmoid(s.a[i]);
            db[i] = dm[i] * s.a[i] * sg;
            da[i] = dm[i] * s.b[i] * sg * (1.0 + s.a[i] * (1.0 - sg));
        }
        let dh2a = mm_back(&s.h2, n, d, &p[l.wg..l.wg + d * f], f, &da, &mut g[l.wg..l.wg + d * f]);
        let dh2b = mm_back(&s.h2, n, d, &p[l.wu..l.wu + d * f], f, &db, &mut g[l.wu..l.wu + d * f]);
        let dh2: Vec<f32> = dh2a.iter().zip(&dh2b).map(|(a, b)| a + b).collect();
        let dx2n = rms_back(&s.x2, n, d, &p[l.g2..l.g2 + d], &s.inv2, &dh2, &mut g[l.g2..l.g2 + d]);
        let dx2: Vec<f32> = dx.iter().zip(&dx2n).map(|(a, b)| a + b).collect();
        // The attention.
        let do_ = mm_back(&s.o, n, d, &p[l.wo..l.wo + d * d], d, &dx2, &mut g[l.wo..l.wo + d * d]);
        let mut dq = vec![0.0; n * d];
        let mut dk = vec![0.0; n * d];
        let mut dv = vec![0.0; n * d];
        for hd in 0..hh {
            for t in 0..n {
                let row = &s.att[(hd * n + t) * n..(hd * n + t + 1) * n];
                let mut da_row = vec![0.0; t + 1];
                let mut dot = 0.0;
                for sidx in 0..=t {
                    let mut acc = 0.0;
                    for j in 0..dh {
                        acc += do_[t * d + hd * dh + j] * s.v[sidx * d + hd * dh + j];
                        dv[sidx * d + hd * dh + j] += row[sidx] * do_[t * d + hd * dh + j];
                    }
                    da_row[sidx] = acc;
                    dot += acc * row[sidx];
                }
                for sidx in 0..=t {
                    let ds = row[sidx] * (da_row[sidx] - dot) * scale;
                    if ds == 0.0 {
                        continue;
                    }
                    for j in 0..dh {
                        dq[t * d + hd * dh + j] += ds * s.k[sidx * d + hd * dh + j];
                        dk[sidx * d + hd * dh + j] += ds * s.q[t * d + hd * dh + j];
                    }
                }
            }
        }
        let dhq = mm_back(&s.h, n, d, &p[l.wq..l.wq + d * d], d, &dq, &mut g[l.wq..l.wq + d * d]);
        let dhk = mm_back(&s.h, n, d, &p[l.wk..l.wk + d * d], d, &dk, &mut g[l.wk..l.wk + d * d]);
        let dhv = mm_back(&s.h, n, d, &p[l.wv..l.wv + d * d], d, &dv, &mut g[l.wv..l.wv + d * d]);
        let dhs: Vec<f32> = (0..n * d).map(|i| dhq[i] + dhk[i] + dhv[i]).collect();
        let dxn = rms_back(&s.x, n, d, &p[l.g1..l.g1 + d], &s.inv1, &dhs, &mut g[l.g1..l.g1 + d]);
        dx = dx2.iter().zip(&dxn).map(|(a, b)| a + b).collect();
    }
    for (t, &tok) in seq.iter().enumerate() {
        for j in 0..d {
            g[lay.emb + tok * d + j] += dx[t * d + j];
            g[lay.pos + t * d + j] += dx[t * d + j];
        }
    }
    loss
}
