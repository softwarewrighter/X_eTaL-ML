//! The synthetic in-context task of Gao and Gao (arXiv 2611.05344,
//! Section 12 and Appendix "Trained model: the gate condition"),
//! reimplemented from the authors' toy/train_toy.py (MIT): 48 words, each
//! in two of eight classes; a sentence is three words sharing exactly one
//! class; 32 labels, redrawn for every prompt. A prompt is the start
//! token, demonstrations `w w w -> label \n`, then the query `w w w ->`.

/// Splitmix64, with the draws the generator needs (std has no RNG).
#[derive(Clone)]
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// An integer in lo..=hi.
    pub fn int(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as usize
    }
    pub fn gauss(&mut self) -> f64 {
        let u1 = loop {
            let u = self.uniform();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
    /// k distinct items of xs, in the order drawn.
    pub fn sample<T: Copy>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        let mut v = xs.to_vec();
        for i in 0..k {
            let j = i + (self.next_u64() % (v.len() - i) as u64) as usize;
            v.swap(i, j);
        }
        v.truncate(k);
        v
    }
}

pub const K: usize = 8;
pub const W: usize = 48;
pub const NL: usize = 32;
pub const BOS: usize = 1;
pub const ARROW: usize = 2;
pub const NEWLINE: usize = 3;
pub const W0: usize = 4;
pub const L0: usize = W0 + W;
pub const VOCAB: usize = L0 + NL;
/// The longest prompt: 4 classes of 3 demonstrations, 6 tokens each, the
/// start, the query and its label.
pub const MAXLEN: usize = 80;

/// Each word's two classes: the 28 pairs of the eight classes shuffled,
/// word w taking pair w mod 28.
pub fn words(seed: u64) -> Vec<(usize, usize)> {
    let mut pairs = vec![];
    for a in 0..K {
        for b in a + 1..K {
            pairs.push((a, b));
        }
    }
    let mut r = Rng(1000 + seed);
    r.shuffle(&mut pairs);
    (0..W).map(|w| pairs[w % pairs.len()]).collect()
}

fn has(p: (usize, usize), c: usize) -> bool {
    p.0 == c || p.1 == c
}

/// Three words whose classes meet in class c alone.
pub fn sentence(r: &mut Rng, c: usize, cls: &[(usize, usize)]) -> Vec<usize> {
    let cand: Vec<usize> = (0..W).filter(|&w| has(cls[w], c)).collect();
    loop {
        let ws = r.sample(&cand, 3);
        let common: Vec<usize> = (0..K).filter(|&k| ws.iter().all(|&w| has(cls[w], k))).collect();
        if common == [c] {
            return ws;
        }
    }
}

/// The kinds of prompt: unique labels with the query's label retrievable
/// ("one"), recurring (each class 1 to 3 demonstrations, a label per
/// class: "two"), and unique labels with a fresh query label ("noise").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    One,
    Two,
    Noise,
}

/// The training mixes: half unique and half recurring ("mix"), recurring
/// only, and half recurring and half noise ("gate").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cond {
    Mix,
    Recurring,
    Gate,
}

impl Cond {
    pub fn name(self) -> &'static str {
        match self {
            Cond::Mix => "mix",
            Cond::Recurring => "recurring",
            Cond::Gate => "gate",
        }
    }
}

/// A prompt: its tokens (the query's label last) and where each label is
/// predicted (the arrow's position, the label).
#[derive(Clone, Debug)]
pub struct Prompt {
    pub seq: Vec<usize>,
    pub targets: Vec<(usize, usize)>,
    pub kind: Kind,
    /// Each demonstration's class, and the query's.
    pub classes: Vec<usize>,
    pub query: usize,
}

pub fn prompt(r: &mut Rng, cond: Cond, cls: &[(usize, usize)]) -> Prompt {
    let kind = match cond {
        Cond::Mix if r.uniform() < 0.5 => Kind::One,
        Cond::Gate if r.uniform() < 0.5 => Kind::Noise,
        _ => Kind::Two,
    };
    kind_prompt(r, kind, cls)
}

pub fn kind_prompt(r: &mut Rng, kind: Kind, cls: &[(usize, usize)]) -> Prompt {
    let all_k: Vec<usize> = (0..K).collect();
    let all_l: Vec<usize> = (0..NL).collect();
    let mut seq = vec![BOS];
    let mut targets = vec![];
    let demo = |seq: &mut Vec<usize>, targets: &mut Vec<(usize, usize)>, r: &mut Rng, c: usize, l: usize, last: bool| {
        seq.extend(sentence(r, c, cls).into_iter().map(|w| W0 + w));
        seq.push(ARROW);
        targets.push((seq.len() - 1, L0 + l));
        seq.push(L0 + l);
        if !last {
            seq.push(NEWLINE);
        }
    };
    match kind {
        Kind::Two => {
            let n = r.int(2, 4);
            let classes = r.sample(&all_k, n);
            let labels = r.sample(&all_l, n);
            let mut demos: Vec<usize> = vec![];
            for i in 0..n {
                for _ in 0..r.int(1, 3) {
                    demos.push(i);
                }
            }
            r.shuffle(&mut demos);
            let q = r.int(0, n - 1);
            for &i in &demos {
                demo(&mut seq, &mut targets, r, classes[i], labels[i], false);
            }
            demo(&mut seq, &mut targets, r, classes[q], labels[q], true);
            Prompt { seq, targets, kind, classes: demos.iter().map(|&i| classes[i]).collect(), query: classes[q] }
        }
        _ => {
            let n = r.int(3, 6);
            let classes = r.sample(&all_k, n);
            let labels = r.sample(&all_l, n);
            let q = r.int(0, n - 1);
            for i in 0..n {
                demo(&mut seq, &mut targets, r, classes[i], labels[i], false);
            }
            let ql = if kind == Kind::One {
                labels[q]
            } else {
                let fresh: Vec<usize> = all_l.iter().copied().filter(|l| !labels.contains(l)).collect();
                fresh[r.int(0, fresh.len() - 1)]
            };
            demo(&mut seq, &mut targets, r, classes[q], ql, true);
            let query = classes[q];
            Prompt { seq, targets, kind, classes, query }
        }
    }
}
