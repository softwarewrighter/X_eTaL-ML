//! Trains the micro-bert demo's tiny BERT and writes its weights into
//! `../data/`, where `../micro-bert.xtl` reads them; also the scores this
//! Rust model gives for a test name with a blank, which the X_eTaL forward
//! pass must match, and how often it fills a blank in names it never saw.
//! The random numbers and the tape autograd are micro-gpt's trainer's
//! (adapted from microgpt-rs, github.com/softwarewrighter/microgpt-rs, MIT,
//! itself a port of Andrej Karpathy's microgpt.py), with tanh for GELU in
//! place of ReLU; the model is BERT's encoder (Devlin et al. 2019): token
//! and position embeddings, layer norm, then L layers of multi-head
//! attention that sees every letter both ways and a GELU feed-forward
//! layer, each added back and layer-normed (post-LN), and a dense layer
//! over the letters. Trained as a masked language model: each name with
//! each letter in turn replaced by the mask token, the loss at the blank.
//! Left out of BERT, on purpose: the [CLS] and [SEP] tokens and segment
//! embeddings (one name, no sentence pairs), biases in attention, and the
//! 80/10/10 masking (always the mask token). The names are makemore's,
//! fetched into work/ by `just micro-bert-data`, not committed.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::ops::{Add, Div, Mul, Neg, Sub};

// Let there be Randomness: Rust's std has no RNG, so we make one (splitmix64 + Box-Muller)
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed) // Let there be order among chaos
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 // [0, 1)
    }

    fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = loop {
            let u = self.uniform();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.uniform();
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            xs.swap(i, j);
        }
    }
}

// Let there be Autograd: a tape of scalar nodes recording the forward pass,
// replayed in reverse to apply the chain rule
#[derive(Clone, Copy)]
struct Parent {
    idx: u32,        // index of the parent node on the tape
    local_grad: f64, // d(this node) / d(parent), captured during the forward pass
}

#[derive(Clone, Copy)]
struct Node {
    data: f64, // scalar value of this node, calculated during the forward pass
    grad: f64, // derivative of the loss w.r.t. this node, calculated in the backward pass
    parents: [Option<Parent>; 2], // every primitive op has at most 2 inputs
}

struct Tape {
    nodes: RefCell<Vec<Node>>,
}

/// A lightweight, copyable handle to one node on the tape.
#[derive(Clone, Copy)]
struct Value<'t> {
    tape: &'t Tape,
    idx: u32,
}

impl Tape {
    fn new() -> Tape {
        Tape { nodes: RefCell::new(Vec::new()) }
    }

    fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// Discard every node past `len` -- frees a whole computation graph in O(1).
    fn truncate(&self, len: usize) {
        self.nodes.borrow_mut().truncate(len);
    }

    /// A leaf node (parameter or constant).
    fn value(&self, data: f64) -> Value<'_> {
        self.push(data, [None, None])
    }

    fn push(&self, data: f64, parents: [Option<Parent>; 2]) -> Value<'_> {
        let mut nodes = self.nodes.borrow_mut();
        let idx = u32::try_from(nodes.len()).expect("tape overflow");
        nodes.push(Node { data, grad: 0.0, parents });
        Value { tape: self, idx }
    }

    /// The tape is appended in forward order, which *is* a topological order of the
    /// computation graph -- so backprop is one reverse sweep, no sorting required.
    fn backward(&self, loss: Value) {
        let mut nodes = self.nodes.borrow_mut();
        nodes[loss.idx as usize].grad = 1.0;
        for i in (0..=loss.idx as usize).rev() {
            let Node { grad, parents, .. } = nodes[i];
            for p in parents.into_iter().flatten() {
                nodes[p.idx as usize].grad += p.local_grad * grad;
            }
        }
    }
}

impl<'t> Value<'t> {
    fn data(self) -> f64 {
        self.tape.nodes.borrow()[self.idx as usize].data
    }

    #[allow(dead_code)] // the training loop reads grads in bulk; this is for the grad-check tests
    fn grad(self) -> f64 {
        self.tape.nodes.borrow()[self.idx as usize].grad
    }

    fn parent(self, local_grad: f64) -> Option<Parent> {
        Some(Parent { idx: self.idx, local_grad })
    }

    fn powf(self, k: f64) -> Value<'t> {
        let a = self.data();
        self.tape.push(a.powf(k), [self.parent(k * a.powf(k - 1.0)), None])
    }

    fn log(self) -> Value<'t> {
        let a = self.data();
        self.tape.push(a.ln(), [self.parent(1.0 / a), None])
    }

    fn exp(self) -> Value<'t> {
        let e = self.data().exp();
        self.tape.push(e, [self.parent(e), None])
    }

    /// tanh, as 2 sigmoid(2a) - 1 (the form X_eTaL's Activations computes).
    fn tanh(self) -> Value<'t> {
        let a = self.data();
        let t = 2.0 / (1.0 + (-2.0 * a).exp()) - 1.0;
        self.tape.push(t, [self.parent(1.0 - t * t), None])
    }
}

impl<'t> Add for Value<'t> {
    type Output = Value<'t>;
    fn add(self, other: Value<'t>) -> Value<'t> {
        self.tape.push(self.data() + other.data(), [self.parent(1.0), other.parent(1.0)])
    }
}

impl<'t> Sub for Value<'t> {
    type Output = Value<'t>;
    fn sub(self, other: Value<'t>) -> Value<'t> {
        self.tape.push(self.data() - other.data(), [self.parent(1.0), other.parent(-1.0)])
    }
}

impl<'t> Mul for Value<'t> {
    type Output = Value<'t>;
    fn mul(self, other: Value<'t>) -> Value<'t> {
        let (a, b) = (self.data(), other.data());
        self.tape.push(a * b, [self.parent(b), other.parent(a)])
    }
}

impl<'t> Div for Value<'t> {
    type Output = Value<'t>;
    fn div(self, other: Value<'t>) -> Value<'t> {
        let (a, b) = (self.data(), other.data());
        self.tape.push(a / b, [self.parent(1.0 / b), other.parent(-a / (b * b))])
    }
}

impl<'t> Neg for Value<'t> {
    type Output = Value<'t>;
    fn neg(self) -> Value<'t> {
        self.tape.push(-self.data(), [self.parent(-1.0), None])
    }
}

// Mixed Value-and-float ops: constants need no node of their own, they just have no gradient
impl<'t> Add<f64> for Value<'t> {
    type Output = Value<'t>;
    fn add(self, c: f64) -> Value<'t> {
        self.tape.push(self.data() + c, [self.parent(1.0), None])
    }
}

impl<'t> Sub<f64> for Value<'t> {
    type Output = Value<'t>;
    fn sub(self, c: f64) -> Value<'t> {
        self.tape.push(self.data() - c, [self.parent(1.0), None])
    }
}

impl<'t> Mul<f64> for Value<'t> {
    type Output = Value<'t>;
    fn mul(self, c: f64) -> Value<'t> {
        self.tape.push(self.data() * c, [self.parent(c), None])
    }
}

impl<'t> Div<f64> for Value<'t> {
    type Output = Value<'t>;
    fn div(self, c: f64) -> Value<'t> {
        self.tape.push(self.data() / c, [self.parent(1.0 / c), None])
    }
}

impl<'t> Mul<Value<'t>> for f64 {
    type Output = Value<'t>;
    fn mul(self, v: Value<'t>) -> Value<'t> {
        v * self
    }
}


// The model's sizes: 16 numbers wide, 2 heads of 8, a feed-forward layer
// 64 wide, 2 layers, names of at most 16 letters.
const D: usize = 16;
const HEADS: usize = 2;
const HD: usize = D / HEADS;
const F: usize = 64;
const L: usize = 2;
const MAXLEN: usize = 16;
const STEPS: usize = 6000;
const BATCH: usize = 4;
const HELD_OUT: usize = 1000;

type Mat<'t> = Vec<Vec<Value<'t>>>;

/// A layer's weights, each matrix a row per input (as X_eTaL multiplies
/// x by it); a dense layer with its bias as one more row (NN's d_ense);
/// a layer norm as two rows, the gain and the bias.
struct Layer<'t> {
    wq: Mat<'t>,
    wk: Mat<'t>,
    wv: Mat<'t>,
    wo: Mat<'t>,
    ln1: Mat<'t>,
    w1: Mat<'t>,
    w2: Mat<'t>,
    ln2: Mat<'t>,
}

struct Model<'t> {
    wte: Mat<'t>,
    wpe: Mat<'t>,
    ln0: Mat<'t>,
    layers: Vec<Layer<'t>>,
    lm: Mat<'t>,
}

fn matrix<'t>(tape: &'t Tape, rng: &mut Rng, rows: usize, cols: usize) -> Mat<'t> {
    (0..rows).map(|_| (0..cols).map(|_| tape.value(rng.gauss(0.0, 0.08))).collect()).collect()
}

/// Weights and a bias row of zeros below them.
fn dense<'t>(tape: &'t Tape, rng: &mut Rng, nin: usize, nout: usize) -> Mat<'t> {
    let mut m = matrix(tape, rng, nin, nout);
    m.push((0..nout).map(|_| tape.value(0.0)).collect());
    m
}

/// A layer norm's gain (ones) and bias (zeros).
fn norm(tape: &Tape) -> Mat<'_> {
    vec![(0..D).map(|_| tape.value(1.0)).collect(), (0..D).map(|_| tape.value(0.0)).collect()]
}

impl<'t> Model<'t> {
    fn init(tape: &'t Tape, rng: &mut Rng, vocab: usize, letters: usize) -> Model<'t> {
        Model {
            wte: matrix(tape, rng, vocab, D),
            wpe: matrix(tape, rng, MAXLEN, D),
            ln0: norm(tape),
            layers: (0..L)
                .map(|_| Layer {
                    wq: matrix(tape, rng, D, D),
                    wk: matrix(tape, rng, D, D),
                    wv: matrix(tape, rng, D, D),
                    wo: matrix(tape, rng, D, D),
                    ln1: norm(tape),
                    w1: dense(tape, rng, D, F),
                    w2: dense(tape, rng, F, D),
                    ln2: norm(tape),
                })
                .collect(),
            lm: dense(tape, rng, D, letters),
        }
    }
}

fn sum<'t>(xs: impl Iterator<Item = Value<'t>>) -> Value<'t> {
    xs.reduce(|a, b| a + b).unwrap()
}

/// x times w (a row per input), rows of x each.
fn times<'t>(x: &[Vec<Value<'t>>], w: &[Vec<Value<'t>>]) -> Mat<'t> {
    x.iter().map(|r| (0..w[0].len()).map(|j| sum(r.iter().zip(w).map(|(a, wr)| *a * wr[j]))).collect()).collect()
}

/// x W + b, the bias the last row of wb (NN's d_ense).
fn dense_of<'t>(x: &[Vec<Value<'t>>], wb: &[Vec<Value<'t>>]) -> Mat<'t> {
    let (w, b) = wb.split_at(wb.len() - 1);
    times(x, w).into_iter().map(|r| r.into_iter().zip(&b[0]).map(|(a, c)| a + *c).collect()).collect()
}

/// Each row standardized (population variance, eps 1e-5, as Norm's
/// s_tandard), times the gain plus the bias.
fn layer_norm<'t>(x: &[Vec<Value<'t>>], gb: &[Vec<Value<'t>>]) -> Mat<'t> {
    x.iter()
        .map(|r| {
            let mean = sum(r.iter().copied()) / D as f64;
            let c: Vec<Value<'t>> = r.iter().map(|a| *a - mean).collect();
            let sd = (sum(c.iter().map(|a| *a * *a)) / D as f64 + 1e-5).powf(0.5);
            c.iter().zip(&gb[0]).zip(&gb[1]).map(|((a, g), b)| *g * (*a / sd) + *b).collect()
        })
        .collect()
}

fn add_rows<'t>(a: &[Vec<Value<'t>>], b: &[Vec<Value<'t>>]) -> Mat<'t> {
    a.iter().zip(b).map(|(r, s)| r.iter().zip(s).map(|(x, y)| *x + *y).collect()).collect()
}

/// GELU, the tanh form (Activations' g_elu).
fn gelu(x: Value) -> Value {
    let u = (x + x * x * x * 0.044715) * 0.7978845608028654;
    x * 0.5 * (u.tanh() + 1.0)
}

fn softmax<'t>(logits: &[Value<'t>]) -> Vec<Value<'t>> {
    let max_val = logits.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value<'t>> = logits.iter().map(|v| (*v - max_val).exp()).collect();
    let total = sum(exps.iter().copied());
    exps.iter().map(|e| *e / total).collect()
}

/// One encoder layer: attention over every letter, both ways, by heads,
/// added back and normed; the feed-forward layer, added back and normed.
/// Also each head's attention weights (for the page).
fn layer<'t>(x: &[Vec<Value<'t>>], w: &Layer<'t>) -> (Mat<'t>, Vec<Vec<Vec<f64>>>) {
    let (q, k, v) = (times(x, &w.wq), times(x, &w.wk), times(x, &w.wv));
    let n = x.len();
    let mut joined: Mat<'t> = vec![Vec::with_capacity(D); n];
    let mut looks = Vec::new();
    for h in 0..HEADS {
        let s = h * HD;
        let mut look = Vec::new();
        for i in 0..n {
            let scores: Vec<Value<'t>> = (0..n).map(|j| sum((0..HD).map(|c| q[i][s + c] * k[j][s + c])) / (HD as f64).sqrt()).collect();
            let a = softmax(&scores);
            look.push(a.iter().map(|p| p.data()).collect());
            for c in 0..HD {
                joined[i].push(sum((0..n).map(|j| a[j] * v[j][s + c])));
            }
        }
        looks.push(look);
    }
    let a = layer_norm(&add_rows(x, &times(&joined, &w.wo)), &w.ln1);
    let hidden: Mat<'t> = dense_of(&a, &w.w1).into_iter().map(|r| r.into_iter().map(gelu).collect()).collect();
    (layer_norm(&add_rows(&a, &dense_of(&hidden, &w.w2)), &w.ln2), looks)
}

/// The scores over the letters at every position of the tokens ids.
fn bert<'t>(m: &Model<'t>, ids: &[usize]) -> (Mat<'t>, Vec<Vec<Vec<Vec<f64>>>>) {
    let x: Mat<'t> = ids.iter().enumerate().map(|(p, &t)| m.wte[t].iter().zip(&m.wpe[p]).map(|(a, b)| *a + *b).collect()).collect();
    let mut h = layer_norm(&x, &m.ln0);
    let mut looks = Vec::new();
    for w in &m.layers {
        let (next, look) = layer(&h, w);
        h = next;
        looks.push(look);
    }
    (dense_of(&h, &m.lm), looks)
}

fn write_rows(path: &str, rows: &[Vec<f64>]) {
    let text: String = rows.iter().map(|r| r.iter().map(|v| format!("{v:?}")).collect::<Vec<_>>().join(" ") + "\n").collect();
    std::fs::write(path, text).unwrap_or_else(|e| panic!("{path}: {e}"));
}

fn data(m: &Mat) -> Vec<Vec<f64>> {
    m.iter().map(|r| r.iter().map(|v| v.data()).collect()).collect()
}

fn main() {
    let mut rng = Rng::new(42);
    let here = env!("CARGO_MANIFEST_DIR");
    let names = format!("{here}/../../../work/names.txt");
    let out = format!("{here}/../data");
    let text = std::fs::read_to_string(&names).unwrap_or_else(|e| panic!("{names}: {e} (run just micro-bert-data first)"));
    let mut docs: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty() && l.len() <= MAXLEN).collect();
    rng.shuffle(&mut docs);
    let (train, held) = docs.split_at(docs.len() - HELD_OUT);
    let letters: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
    let stoi: HashMap<char, usize> = letters.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let mask = letters.len();
    let tape = Tape::new();
    let model = Model::init(&tape, &mut rng, letters.len() + 1, letters.len());
    let params = tape.len();
    eprintln!("{} names ({} held out), {} letters and the mask, {params} weights", docs.len(), held.len(), letters.len());

    let (lr, beta1, beta2, eps) = (0.01, 0.85, 0.99, 1e-8);
    let (mut m1, mut m2) = (vec![0.0; params], vec![0.0; params]);
    let mut recent = 0.0;
    for step in 0..STEPS {
        // BATCH names, each with every letter blank in turn.
        let mut losses = Vec::new();
        for b in 0..BATCH {
            let ids: Vec<usize> = train[(step * BATCH + b) % train.len()].chars().map(|c| stoi[&c]).collect();
            for p in 0..ids.len() {
                let mut blank = ids.clone();
                blank[p] = mask;
                let (logits, _) = bert(&model, &blank);
                losses.push(-softmax(&logits[p])[ids[p]].log());
            }
        }
        let count = losses.len();
        let loss = sum(losses.into_iter()) / count as f64;
        recent = if step == 0 { loss.data() } else { 0.99 * recent + 0.01 * loss.data() };
        tape.backward(loss);
        let lr_t = lr * (1.0 - step as f64 / STEPS as f64);
        {
            let mut nodes = tape.nodes.borrow_mut();
            for i in 0..params {
                let g = nodes[i].grad;
                m1[i] = beta1 * m1[i] + (1.0 - beta1) * g;
                m2[i] = beta2 * m2[i] + (1.0 - beta2) * g * g;
                let a = m1[i] / (1.0 - beta1.powi(step as i32 + 1));
                let b = m2[i] / (1.0 - beta2.powi(step as i32 + 1));
                nodes[i].data -= lr_t * a / (b.sqrt() + eps);
                nodes[i].grad = 0.0;
            }
        }
        tape.truncate(params);
        if step % 1000 == 999 {
            eprintln!("step {}: loss about {recent:.3}", step + 1);
        }
    }

    // How often it fills a blank in names it never saw: every letter of
    // the held-out names in turn, the likeliest letter (the first on a
    // tie); beside guessing the commonest letter every time.
    let mut count = vec![0usize; letters.len()];
    for d in train {
        for c in d.chars() {
            count[stoi[&c]] += 1;
        }
    }
    let common = (0..letters.len()).max_by_key(|&i| (count[i], usize::MAX - i)).unwrap();
    let (mut right, mut three, mut total, mut guess) = (0usize, 0usize, 0usize, 0usize);
    for d in held {
        let ids: Vec<usize> = d.chars().map(|c| stoi[&c]).collect();
        for p in 0..ids.len() {
            let mut blank = ids.clone();
            blank[p] = mask;
            let (logits, _) = bert(&model, &blank);
            let s: Vec<f64> = logits[p].iter().map(|v| v.data()).collect();
            let best = (0..s.len()).fold(0, |b, j| if s[j] > s[b] { j } else { b });
            right += (best == ids[p]) as usize;
            three += (s.iter().filter(|&&x| x > s[ids[p]]).count() < 3) as usize;
            guess += (common == ids[p]) as usize;
            total += 1;
            tape.truncate(params);
        }
    }
    eprintln!("trained {STEPS} steps of {BATCH} names; held out: {right} of {total} blanks filled right, {three} in its top 3, the commonest letter ({}) {guess}", letters[common]);

    std::fs::create_dir_all(&out).expect("data/");
    let p = |name: &str| format!("{out}/{name}.txt");
    std::fs::write(p("chars"), letters.iter().collect::<String>() + "\n").expect("chars.txt");
    write_rows(&p("wte"), &data(&model.wte));
    write_rows(&p("wpe"), &data(&model.wpe));
    write_rows(&p("ln0"), &data(&model.ln0));
    write_rows(&p("lm"), &data(&model.lm));
    // Each layer's matrices one after another (L blocks of rows).
    let stack = |f: &dyn Fn(&Layer) -> Vec<Vec<f64>>| -> Vec<Vec<f64>> { model.layers.iter().flat_map(|w| f(w)).collect() };
    write_rows(&p("wq"), &stack(&|w| data(&w.wq)));
    write_rows(&p("wk"), &stack(&|w| data(&w.wk)));
    write_rows(&p("wv"), &stack(&|w| data(&w.wv)));
    write_rows(&p("wo"), &stack(&|w| data(&w.wo)));
    write_rows(&p("ln1"), &stack(&|w| data(&w.ln1)));
    write_rows(&p("w1"), &stack(&|w| data(&w.w1)));
    write_rows(&p("w2"), &stack(&|w| data(&w.w2)));
    write_rows(&p("ln2"), &stack(&|w| data(&w.ln2)));
    std::fs::write(p("accuracy"), format!("{right} {three} {total} {guess}\n")).expect("accuracy.txt");
    // The scores for "emma" with its third letter blank, every position,
    // and the attention of each layer's heads, for the X_eTaL checks.
    let mut ids: Vec<usize> = "emma".chars().map(|c| stoi[&c]).collect();
    ids[2] = mask;
    let (logits, looks) = bert(&model, &ids);
    write_rows(&p("expected"), &data(&logits));
    write_rows(&p("expected-looks"), &looks.iter().flat_map(|l| l.iter().map(|h| h.iter().flatten().copied().collect())).collect::<Vec<Vec<f64>>>());
    tape.truncate(params);
    eprintln!("wrote {out}/: the letters, the weights, accuracy.txt, expected.txt, expected-looks.txt");
}
