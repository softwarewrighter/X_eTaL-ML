//! Trains the micro-gpt demo's tiny GPT and writes its weights into
//! `../data/`, where `../micro-gpt.xtl` reads them; also the logits this
//! Rust model gives for a test name, which the X_eTaL forward pass must
//! match. Adapted from microgpt-rs (github.com/softwarewrighter/microgpt-rs,
//! MIT, the same author), itself a port of Andrej Karpathy's microgpt.py:
//! the same model, training and seed; only `main` is new (where the names
//! come from, what is written). The names are makemore's (32k baby names),
//! fetched into work/ by `just micro-gpt-data`, not committed.
//!
//! The most atomic way to train and run inference for a GPT in pure, dependency-free Rust.
//! This file is the complete algorithm. Everything else is just efficiency.
//!
//! A port of Andrej Karpathy's microgpt.py. One deliberate departure: instead of a
//! garbage-collected object graph, autograd is a *tape* (arena) of nodes. Nodes are
//! appended in forward order, which is already a topological order, so the backward
//! pass is a single reverse sweep -- no DFS, no visited set.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::io::Write;
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

    /// Sample an index in proportion to the given weights (need not sum to 1).
    fn choices(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.uniform() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
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

    fn relu(self) -> Value<'t> {
        let a = self.data();
        self.tape.push(a.max(0.0), [self.parent(if a > 0.0 { 1.0 } else { 0.0 }), None])
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

// Initialize the parameters, to store the knowledge of the model
const N_LAYER: usize = 1; // depth of the transformer neural network (number of layers)
const N_EMBD: usize = 16; // width of the network (embedding dimension)
const BLOCK_SIZE: usize = 16; // maximum context length of the attention window
const N_HEAD: usize = 4; // number of attention heads
const HEAD_DIM: usize = N_EMBD / N_HEAD; // derived dimension of each head

type Mat<'t> = Vec<Vec<Value<'t>>>;

struct LayerWeights<'t> {
    attn_wq: Mat<'t>,
    attn_wk: Mat<'t>,
    attn_wv: Mat<'t>,
    attn_wo: Mat<'t>,
    mlp_fc1: Mat<'t>,
    mlp_fc2: Mat<'t>,
}

struct StateDict<'t> {
    wte: Mat<'t>,
    wpe: Mat<'t>,
    lm_head: Mat<'t>,
    layers: Vec<LayerWeights<'t>>,
}

fn matrix<'t>(tape: &'t Tape, rng: &mut Rng, nout: usize, nin: usize) -> Mat<'t> {
    let std = 0.08;
    (0..nout).map(|_| (0..nin).map(|_| tape.value(rng.gauss(0.0, std))).collect()).collect()
}

impl<'t> StateDict<'t> {
    fn init(tape: &'t Tape, rng: &mut Rng, vocab_size: usize) -> StateDict<'t> {
        StateDict {
            wte: matrix(tape, rng, vocab_size, N_EMBD),
            wpe: matrix(tape, rng, BLOCK_SIZE, N_EMBD),
            lm_head: matrix(tape, rng, vocab_size, N_EMBD),
            layers: (0..N_LAYER)
                .map(|_| LayerWeights {
                    attn_wq: matrix(tape, rng, N_EMBD, N_EMBD),
                    attn_wk: matrix(tape, rng, N_EMBD, N_EMBD),
                    attn_wv: matrix(tape, rng, N_EMBD, N_EMBD),
                    attn_wo: matrix(tape, rng, N_EMBD, N_EMBD),
                    mlp_fc1: matrix(tape, rng, 4 * N_EMBD, N_EMBD),
                    mlp_fc2: matrix(tape, rng, N_EMBD, 4 * N_EMBD),
                })
                .collect(),
        }
    }
}

// Define the model architecture: a function mapping tokens and parameters to logits over what comes next
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
fn linear<'t>(x: &[Value<'t>], w: &[Vec<Value<'t>>]) -> Vec<Value<'t>> {
    w.iter()
        .map(|row| row.iter().zip(x).map(|(wi, xi)| *wi * *xi).reduce(|a, b| a + b).unwrap())
        .collect()
}

fn softmax<'t>(logits: &[Value<'t>]) -> Vec<Value<'t>> {
    let max_val = logits.iter().map(|v| v.data()).fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<Value<'t>> = logits.iter().map(|v| (*v - max_val).exp()).collect();
    let total = exps.iter().copied().reduce(|a, b| a + b).unwrap();
    exps.iter().map(|e| *e / total).collect()
}

fn rmsnorm<'t>(x: &[Value<'t>]) -> Vec<Value<'t>> {
    let ms = x.iter().map(|xi| *xi * *xi).reduce(|a, b| a + b).unwrap() / x.len() as f64;
    let scale = (ms + 1e-5).powf(-0.5);
    x.iter().map(|xi| *xi * scale).collect()
}

type KvCache<'t> = Vec<Vec<Vec<Value<'t>>>>; // [layer][position][n_embd]

fn gpt<'t>(
    sd: &StateDict<'t>,
    token_id: usize,
    pos_id: usize,
    keys: &mut KvCache<'t>,
    values: &mut KvCache<'t>,
) -> Vec<Value<'t>> {
    let tok_emb = &sd.wte[token_id]; // token embedding
    let pos_emb = &sd.wpe[pos_id]; // position embedding
    let mut x: Vec<Value<'t>> = tok_emb.iter().zip(pos_emb).map(|(t, p)| *t + *p).collect();
    x = rmsnorm(&x); // note: not redundant due to backward pass via the residual connection

    for (li, layer) in sd.layers.iter().enumerate() {
        // 1) Multi-head Attention block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &layer.attn_wq);
        let k = linear(&x, &layer.attn_wk);
        let v = linear(&x, &layer.attn_wv);
        keys[li].push(k);
        values[li].push(v);
        let t_len = keys[li].len();
        let mut x_attn = Vec::with_capacity(N_EMBD);
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let q_h = &q[hs..hs + HEAD_DIM];
            let attn_logits: Vec<Value<'t>> = (0..t_len)
                .map(|t| {
                    let k_h = &keys[li][t][hs..hs + HEAD_DIM];
                    let dot = (0..HEAD_DIM).map(|j| q_h[j] * k_h[j]).reduce(|a, b| a + b).unwrap();
                    dot / (HEAD_DIM as f64).sqrt()
                })
                .collect();
            let attn_weights = softmax(&attn_logits);
            for j in 0..HEAD_DIM {
                let head_out = (0..t_len)
                    .map(|t| attn_weights[t] * values[li][t][hs + j])
                    .reduce(|a, b| a + b)
                    .unwrap();
                x_attn.push(head_out);
            }
        }
        x = linear(&x_attn, &layer.attn_wo);
        x = x.iter().zip(&x_residual).map(|(a, b)| *a + *b).collect();
        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &layer.mlp_fc1);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &layer.mlp_fc2);
        x = x.iter().zip(&x_residual).map(|(a, b)| *a + *b).collect();
    }

    linear(&x, &sd.lm_head) // logits
}

fn write_mat(dir: &str, name: &str, m: &Mat) {
    let text: String = m.iter().map(|row| row.iter().map(|v| format!("{:?}", v.data())).collect::<Vec<_>>().join(" ") + "\n").collect();
    std::fs::write(format!("{dir}/{name}.txt"), text).unwrap_or_else(|e| panic!("{dir}/{name}.txt: {e}"));
}

fn main() {
    let mut rng = Rng::new(42);
    let here = env!("CARGO_MANIFEST_DIR");
    let names = format!("{here}/../../../work/names.txt");
    let data = format!("{here}/../data");
    let text = std::fs::read_to_string(&names).unwrap_or_else(|e| panic!("{names}: {e} (run just micro-gpt-data first)"));
    let mut docs: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    rng.shuffle(&mut docs);
    eprintln!("num docs: {}", docs.len());

    let uchars: Vec<char> = docs.iter().flat_map(|d| d.chars()).collect::<BTreeSet<_>>().into_iter().collect();
    let stoi: HashMap<char, usize> = uchars.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let bos = uchars.len();
    let vocab_size = uchars.len() + 1;

    let tape = Tape::new();
    let sd = StateDict::init(&tape, &mut rng, vocab_size);
    let num_params = tape.len();
    eprintln!("vocab size: {vocab_size}, num params: {num_params}");

    let (learning_rate, beta1, beta2, eps_adam) = (0.01, 0.85, 0.99, 1e-8);
    let mut m = vec![0.0; num_params];
    let mut v = vec![0.0; num_params];
    let num_steps = 1000;
    let mut last = 0.0;
    for step in 0..num_steps {
        let doc = docs[step % docs.len()];
        let mut tokens = vec![bos];
        tokens.extend(doc.chars().map(|c| stoi[&c]));
        tokens.push(bos);
        let n = BLOCK_SIZE.min(tokens.len() - 1);
        let mut keys: KvCache = vec![Vec::new(); N_LAYER];
        let mut values: KvCache = vec![Vec::new(); N_LAYER];
        let mut losses = Vec::with_capacity(n);
        for pos_id in 0..n {
            let (token_id, target_id) = (tokens[pos_id], tokens[pos_id + 1]);
            let logits = gpt(&sd, token_id, pos_id, &mut keys, &mut values);
            let probs = softmax(&logits);
            losses.push(-probs[target_id].log());
        }
        let loss = (1.0 / n as f64) * losses.into_iter().reduce(|a, b| a + b).unwrap();
        last = 0.98 * last + 0.02 * loss.data();
        tape.backward(loss);
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64);
        {
            let mut nodes = tape.nodes.borrow_mut();
            for i in 0..num_params {
                let g = nodes[i].grad;
                m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
                let m_hat = m[i] / (1.0 - beta1.powi(step as i32 + 1));
                let v_hat = v[i] / (1.0 - beta2.powi(step as i32 + 1));
                nodes[i].data -= lr_t * m_hat / (v_hat.sqrt() + eps_adam);
                nodes[i].grad = 0.0;
            }
        }
        drop((keys, values));
        tape.truncate(num_params);
    }
    eprintln!("trained {num_steps} steps, recent loss about {last:.3}");

    // The weights, each matrix a row per output (as the model keeps them).
    std::fs::create_dir_all(&data).expect("data/");
    std::fs::write(format!("{data}/chars.txt"), uchars.iter().collect::<String>() + "\n").expect("chars.txt");
    write_mat(&data, "wte", &sd.wte);
    write_mat(&data, "wpe", &sd.wpe);
    write_mat(&data, "lm_head", &sd.lm_head);
    let l = &sd.layers[0];
    for (name, mm) in [("wq", &l.attn_wq), ("wk", &l.attn_wk), ("wv", &l.attn_wv), ("wo", &l.attn_wo), ("fc1", &l.mlp_fc1), ("fc2", &l.mlp_fc2)] {
        write_mat(&data, name, mm);
    }
    // The logits for BOS then "emma", position by position, for the X_eTaL test.
    let test: Vec<usize> = std::iter::once(bos).chain("emma".chars().map(|c| stoi[&c])).collect();
    let mut keys: KvCache = vec![Vec::new(); N_LAYER];
    let mut values: KvCache = vec![Vec::new(); N_LAYER];
    let rows: String = test.iter().enumerate().map(|(pos, &t)| gpt(&sd, t, pos, &mut keys, &mut values).iter().map(|v| format!("{:?}", v.data())).collect::<Vec<_>>().join(" ") + "\n").collect();
    std::fs::write(format!("{data}/expected.txt"), rows).expect("expected.txt");
    drop((keys, values));
    tape.truncate(num_params);

    // The batch program's data (batch.xtl): the first 1,000 names of the
    // file as rows of token ids (the start token, the letters, the end
    // token, numbered from 1 as X_eTaL selects them, then zeros to 17),
    // the mean loss per token this (Rust) forward pass gives them, and
    // how many next tokens it predicts right (the likeliest, the first
    // on a tie).
    let first: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).take(1000).collect();
    let (mut rows, mut total, mut count, mut right) = (String::new(), 0.0, 0usize, 0usize);
    for name in &first {
        let mut tokens = vec![bos];
        tokens.extend(name.chars().map(|c| stoi[&c]));
        tokens.push(bos);
        let n = BLOCK_SIZE.min(tokens.len() - 1);
        let mut keys: KvCache = vec![Vec::new(); N_LAYER];
        let mut values: KvCache = vec![Vec::new(); N_LAYER];
        for pos in 0..n {
            let probs: Vec<f64> = softmax(&gpt(&sd, tokens[pos], pos, &mut keys, &mut values)).iter().map(|p| p.data()).collect();
            total -= probs[tokens[pos + 1]].ln();
            count += 1;
            let mut best = 0;
            for j in 1..probs.len() {
                if probs[j] > probs[best] {
                    best = j;
                }
            }
            right += (best == tokens[pos + 1]) as usize;
        }
        drop((keys, values));
        tape.truncate(num_params);
        let mut ids: Vec<String> = tokens.iter().map(|t| (t + 1).to_string()).collect();
        ids.resize(17, "0".into());
        rows += &(ids.join(" ") + "\n");
    }
    std::fs::write(format!("{data}/names.txt"), rows).expect("names.txt");
    std::fs::write(format!("{data}/names-loss.txt"), format!("{:?} {right}\n", total / count as f64)).expect("names-loss.txt");
    eprintln!("wrote {data}/: chars, the weights, expected.txt; the first {} names ({count} tokens): mean loss {:.4}, {right} next tokens right (names.txt, names-loss.txt)", first.len(), total / count as f64);
}
