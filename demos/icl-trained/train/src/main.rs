//! Trains the icl-trained demo's transformers on Gao and Gao's synthetic
//! in-context task (arXiv 2611.05344, Section 12; reimplemented from the
//! authors' toy/train_toy.py, MIT) under one of its three training mixes,
//! and writes the weights and held-out prompts into ../data/<mix>/.
//!
//! usage: icl-trained-train [--cond mix|recurring|gate] [--seed N]
//!        [--steps N] [--d N] [--layers N] [--heads N] [--ffn N]
//!        [--bs N] [--threads N] [--out DIR] [--check]
//!
//! The paper's optimizer is kept: AdamW (lr 1e-3, betas 0.9 and 0.95,
//! weight decay 0.1), 200 warmup steps then a linear decay to 5%, the
//! gradient clipped to norm 1, 64 prompts a step. The model is smaller.

mod model;
mod task;

use model::{backward, forward, Config, Layout};
use std::io::Write;
use task::{kind_prompt, prompt, words, Cond, Kind, Prompt, Rng, L0, MAXLEN, VOCAB};

struct Args {
    cond: Cond,
    seed: u64,
    steps: usize,
    c: Config,
    bs: usize,
    threads: usize,
    out: Option<String>,
    check: bool,
    every: usize,
    export: Option<(String, String)>,
    prompts: Option<String>,
}

fn args() -> Args {
    let mut a = Args {
        cond: Cond::Mix,
        seed: 0,
        steps: 20000,
        c: Config { d: 32, layers: 4, heads: 4, ffn: 64, vocab: VOCAB, maxlen: MAXLEN },
        bs: 64,
        threads: 1,
        out: None,
        check: false,
        every: 250,
        export: None,
        prompts: None,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().unwrap_or_default();
        let num = || val().parse::<usize>().expect("a number");
        match v[i].as_str() {
            "--cond" => {
                a.cond = match val().as_str() {
                    "mix" => Cond::Mix,
                    "recurring" => Cond::Recurring,
                    "gate" => Cond::Gate,
                    o => panic!("no mix {o}"),
                }
            }
            "--seed" => a.seed = num() as u64,
            "--steps" => a.steps = num(),
            "--d" => a.c.d = num(),
            "--layers" => a.c.layers = num(),
            "--heads" => a.c.heads = num(),
            "--ffn" => a.c.ffn = num(),
            "--bs" => a.bs = num(),
            "--threads" => a.threads = num(),
            "--every" => a.every = num(),
            "--out" => a.out = Some(val()),
            "--export" => {
                a.export = Some((val(), v.get(i + 2).cloned().unwrap_or_default()));
                i += 1;
            }
            "--prompts" => a.prompts = Some(val()),
            "--check" => {
                a.check = true;
                i -= 1;
            }
            o => panic!("unknown argument {o}"),
        }
        i += 2;
    }
    a
}

/// The query's margin: its label's logit less the best other label shown
/// in the prompt.
pub fn margin(logits: &[f32], p: &Prompt) -> f32 {
    let (t, y) = *p.targets.last().unwrap();
    let row = &logits[t * VOCAB..(t + 1) * VOCAB];
    let mut shown: Vec<usize> = p.seq.iter().copied().filter(|&x| x >= L0 && x != y).collect();
    shown.sort();
    shown.dedup();
    let best = shown.iter().map(|&l| row[l]).fold(f32::MIN, f32::max);
    row[y] - best
}

fn grads(lay: &Layout, p: &[f32], batch: &[Prompt], threads: usize) -> (Vec<f32>, f32) {
    let total: usize = batch.iter().map(|b| b.targets.len()).sum();
    let w = 1.0 / total as f32;
    let chunk = batch.len().div_ceil(threads);
    let parts: Vec<(Vec<f32>, f32)> = std::thread::scope(|s| {
        let hs: Vec<_> = batch
            .chunks(chunk)
            .map(|ch| {
                s.spawn(move || {
                    let mut g = vec![0.0; lay.size];
                    let mut loss = 0.0;
                    for b in ch {
                        let input = &b.seq[..b.seq.len() - 1];
                        let fw = forward(lay, p, input);
                        loss += backward(lay, p, input, &fw, &b.targets, w, &mut g);
                    }
                    (g, loss)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut g = vec![0.0; lay.size];
    let mut loss = 0.0;
    for (pg, l) in parts {
        for (a, b) in g.iter_mut().zip(&pg) {
            *a += b;
        }
        loss += l;
    }
    (g, loss / total as f32)
}

fn check(lay: &Layout, p: &mut [f32], batch: &[Prompt]) {
    let (g, _) = grads(lay, p, batch, 1);
    let loss = |p: &[f32]| {
        let total: usize = batch.iter().map(|b| b.targets.len()).sum();
        let mut l = 0.0f64;
        for b in batch {
            let input = &b.seq[..b.seq.len() - 1];
            let fw = forward(lay, p, input);
            let mut tmp = vec![0.0; lay.size];
            l += backward(lay, p, input, &fw, &b.targets, 0.0, &mut tmp) as f64;
        }
        l / total as f64
    };
    let mut r = Rng(5);
    let mut worst = 0.0f64;
    for _ in 0..40 {
        let i = r.int(0, lay.size - 1);
        let h = 1e-2f32;
        let old = p[i];
        p[i] = old + h;
        let up = loss(p);
        p[i] = old - h;
        let dn = loss(p);
        p[i] = old;
        let fd = (up - dn) / (2.0 * h as f64);
        let err = (fd - g[i] as f64).abs() / (fd.abs() + g[i].abs() as f64 + 1e-4);
        worst = worst.max(err);
        println!("param {i}: backward {:.6e} differences {:.6e}", g[i], fd);
    }
    println!("worst relative error {worst:.3e}");
}

fn main() {
    let a = args();
    let lay = Layout::new(a.c);
    let cls = words(0);
    let mut r = Rng(0x1C1 + a.seed * 7919);
    let mut p: Vec<f32> = (0..lay.size).map(|_| (0.02 * r.gauss()) as f32).collect();
    for g in lay.gains() {
        for j in 0..a.c.d {
            p[g + j] = 1.0;
        }
    }
    if let Some((run, to)) = &a.export {
        export(run, to, a.c);
        return;
    }
    if let Some(to) = &a.prompts {
        prompts(to, 40);
        return;
    }
    let mut data = Rng(0xDA7A + a.seed);
    if a.check {
        let batch: Vec<Prompt> = (0..3).map(|_| prompt(&mut data, Cond::Mix, &cls)).collect();
        check(&lay, &mut p, &batch);
        return;
    }
    // Held-out prompts of each kind, from a stream of their own.
    let mut held = Rng(0x4E1D);
    let eval: Vec<(Kind, Vec<Prompt>)> = [Kind::One, Kind::Two, Kind::Noise].iter().map(|&k| (k, (0..96).map(|_| kind_prompt(&mut held, k, &cls)).collect())).collect();
    eprintln!("{} weights; mix {}, seed {}, {} steps", lay.size, a.cond.name(), a.seed, a.steps);
    let (mut m, mut v) = (vec![0.0f32; lay.size], vec![0.0f32; lay.size]);
    let (b1, b2, lr0, wd) = (0.9f32, 0.95f32, 1e-3f32, 0.1f32);
    let t0 = std::time::Instant::now();
    let mut log = String::new();
    for step in 0..a.steps {
        let batch: Vec<Prompt> = (0..a.bs).map(|_| prompt(&mut data, a.cond, &cls)).collect();
        let (mut g, loss) = grads(&lay, &p, &batch, a.threads);
        let norm = g.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1.0 {
            g.iter_mut().for_each(|x| *x /= norm);
        }
        let lr = lr0 * ((step + 1) as f32 / 200.0).min(1.0) * (1.0 - step as f32 / a.steps as f32).max(0.05);
        let (c1, c2) = (1.0 - b1.powi(step as i32 + 1), 1.0 - b2.powi(step as i32 + 1));
        for i in 0..lay.size {
            m[i] = b1 * m[i] + (1.0 - b1) * g[i];
            v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
            p[i] -= lr * (wd * p[i] + (m[i] / c1) / ((v[i] / c2).sqrt() + 1e-8));
        }
        if step % a.every == 0 || step + 1 == a.steps {
            let ms: Vec<String> = eval
                .iter()
                .map(|(k, ps)| {
                    let mean = ps.iter().map(|q| margin(&forward(&lay, &p, &q.seq[..q.seq.len() - 1]).logits, q)).sum::<f32>() / ps.len() as f32;
                    format!("{k:?} {mean:.3}")
                })
                .collect();
            let line = format!("{step} {loss:.4} {} {:.0}s", ms.join(" "), t0.elapsed().as_secs_f32());
            eprintln!("{line}");
            log.push_str(&line);
            log.push('\n');
        }
    }
    if let Some(out) = a.out {
        std::fs::create_dir_all(&out).unwrap();
        let mut f = std::fs::File::create(format!("{out}/weights.bin")).unwrap();
        for x in &p {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        std::fs::write(format!("{out}/log.txt"), log).unwrap();
        std::fs::write(format!("{out}/config.txt"), format!("{:?} {} {}\n", a.c, a.cond.name(), a.seed)).unwrap();
    }
}

/// Numbers, a row a line, rounded to 4 decimals (the model the demo
/// ships is the rounded one).
fn rows(v: &[f32], width: usize) -> String {
    v.chunks(width).map(|r| r.iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join("\n") + "\n"
}

/// A prompt as a row: its length, then its tokens padded with 0 to MAXLEN.
fn prompt_row(p: &Prompt) -> String {
    let mut v = vec![p.seq.len()];
    v.extend(&p.seq);
    v.resize(MAXLEN + 1, 0);
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

/// The query sentence of a recurring prompt replaced by a sentence of
/// another class shown (the paper's class-flip twin); None if it has one class.
fn flipped(p: &Prompt, r: &mut Rng, cls: &[(usize, usize)]) -> Option<Prompt> {
    let mut others: Vec<usize> = p.classes.iter().copied().filter(|&c| c != p.query).collect();
    others.sort();
    others.dedup();
    if others.is_empty() {
        return None;
    }
    let c = others[r.int(0, others.len() - 1)];
    let n = p.seq.len();
    let mut seq = p.seq[..n - 5].to_vec();
    seq.extend(task::sentence(r, c, cls).into_iter().map(|w| task::W0 + w));
    seq.extend(&p.seq[n - 2..]);
    Some(Prompt { seq, query: c, ..p.clone() })
}

/// Write a trained run (`run`/weights.bin, its config) into `to`: the
/// weights rounded, its log, its margins on 160 held-out prompts of each
/// kind and the query's logits for the first three of each (both with the
/// rounded weights).
pub fn export(run: &str, to: &str, c: Config) {
    let lay = Layout::new(c);
    let bytes = std::fs::read(format!("{run}/weights.bin")).expect("weights.bin");
    let p: Vec<f32> = bytes.chunks(4).map(|b| (f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * 1e4).round() / 1e4).collect();
    assert_eq!(p.len(), lay.size);
    std::fs::create_dir_all(to).unwrap();
    let (d, f, v) = (c.d, c.ffn, c.vocab);
    let w = |name: &str, s: String| std::fs::write(format!("{to}/{name}.txt"), s).unwrap();
    w("emb", rows(&p[lay.emb..lay.emb + v * d], d));
    w("pos", rows(&p[lay.pos..lay.pos + c.maxlen * d], d));
    let cat = |at: fn(&model::Layer) -> usize, n: usize| lay.layers.iter().flat_map(|l| p[at(l)..at(l) + n].to_vec()).collect::<Vec<f32>>();
    w("g1", rows(&cat(|l| l.g1, d), d));
    w("g2", rows(&cat(|l| l.g2, d), d));
    w("wq", rows(&cat(|l| l.wq, d * d), d));
    w("wk", rows(&cat(|l| l.wk, d * d), d));
    w("wv", rows(&cat(|l| l.wv, d * d), d));
    w("wo", rows(&cat(|l| l.wo, d * d), d));
    w("wg", rows(&cat(|l| l.wg, d * f), f));
    w("wu", rows(&cat(|l| l.wu, d * f), f));
    w("wd", rows(&cat(|l| l.wd, f * d), d));
    w("gf", rows(&p[lay.gf..lay.gf + d], d));
    w("un", rows(&p[lay.un..lay.un + d * v], v));
    std::fs::copy(format!("{run}/log.txt"), format!("{to}/log.txt")).unwrap();
    let cls = words(0);
    let mut held = Rng(0x4E1D);
    let mut summary = String::new();
    let mut expected = String::new();
    for k in [Kind::One, Kind::Two, Kind::Noise] {
        let ps: Vec<Prompt> = (0..160).map(|_| kind_prompt(&mut held, k, &cls)).collect();
        let ms: Vec<f32> = ps.iter().map(|q| margin(&forward(&lay, &p, &q.seq[..q.seq.len() - 1]).logits, q)).collect();
        summary.push_str(&format!("{:.4}\n", ms.iter().sum::<f32>() / ms.len() as f32));
    }
    // The query's logits for the first three unique and recurring analysis prompts.
    let (one, two, _) = analysis(40);
    for q in one[..3].iter().chain(&two[..3]) {
        let input = &q.seq[..q.seq.len() - 1];
        let lg = forward(&lay, &p, input).logits;
        let t = input.len() - 1;
        expected.push_str(&rows(&lg[t * v..(t + 1) * v], v));
    }
    w("margins", summary);
    w("expected", expected);
}

/// The analysis prompts: unique, recurring, and the recurring ones' twins.
pub fn analysis(n: usize) -> (Vec<Prompt>, Vec<Prompt>, Vec<Prompt>) {
    let cls = words(0);
    let mut r = Rng(0xA11CE);
    let one: Vec<Prompt> = (0..n).map(|_| kind_prompt(&mut r, Kind::One, &cls)).collect();
    let mut two = vec![];
    let mut flip = vec![];
    while two.len() < n {
        let q = kind_prompt(&mut r, Kind::Two, &cls);
        if let Some(fq) = flipped(&q, &mut r, &cls) {
            two.push(q);
            flip.push(fq);
        }
    }
    (one, two, flip)
}

/// The prompts the X_eTaL analysis reads (shared by every model): unique
/// and recurring prompts, and each recurring prompt's class-flip twin;
/// and each word's two classes.
pub fn prompts(to: &str, n: usize) {
    let cls = words(0);
    let (one, two, flip) = analysis(n);
    let lines = |v: &[Prompt]| v.iter().map(prompt_row).collect::<Vec<_>>().join("\n") + "\n";
    std::fs::create_dir_all(to).unwrap();
    std::fs::write(format!("{to}/unique.txt"), lines(&one)).unwrap();
    std::fs::write(format!("{to}/recurring.txt"), lines(&two)).unwrap();
    std::fs::write(format!("{to}/flipped.txt"), lines(&flip)).unwrap();
    std::fs::write(format!("{to}/words.txt"), cls.iter().map(|(a, b)| format!("{} {}", a + 1, b + 1)).collect::<Vec<_>>().join("\n") + "\n").unwrap();
}
