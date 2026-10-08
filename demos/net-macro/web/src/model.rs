//! The page's state: the spec in the box, and what the macro made of
//! it. A spec that is one of the program's networks runs that network;
//! any other is expanded and counted. Any spec from 2 inputs to 3
//! softmax can also be trained here, from random weights, by the step
//! the macro writes for it.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{clean, networks, run, step_written, train, trainable, typed, After, Network, Trained, Typed};

/// Training steps per run (a frame).
pub const CHUNK: usize = 10;
/// Where training stops by itself.
pub const LIMIT: usize = 400;

#[derive(Clone, Debug, PartialEq)]
pub enum Shown {
    Trained(Network, Rc<Trained>),
    Typed(Typed),
    /// The macro refused the spec: its message.
    Refused(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub spec: String,
    pub shown: Shown,
    pub ms: f64,
    pub training: Option<Training>,
    pub notice: Option<String>,
}

/// The spec being trained in the page.
#[derive(Clone, Debug, PartialEq)]
pub struct Training {
    pub sizes: Vec<usize>,
    /// The training step the macro wrote (`xetal expand`).
    pub step: Rc<String>,
    pub after: Rc<After>,
    pub steps: usize,
    /// (steps, loss, share right) after each run.
    pub history: Vec<(usize, f64, f64)>,
    pub playing: bool,
    /// Milliseconds per step, the last run's.
    pub ms_per_step: f64,
}

pub enum Action {
    Spec(String),
    /// Train the spec from its random starting weights (again).
    Train,
    Tick,
    Play(bool),
}

impl Model {
    pub fn new() -> Self {
        Self::of(networks().last().map(|n| n.spec.clone()).unwrap_or_default())
    }

    fn of(spec: String) -> Self {
        let t = now();
        let spec = clean(&spec);
        let words = |s: &str| s.split_whitespace().map(str::to_string).collect::<Vec<_>>();
        let shown = match networks().into_iter().find(|n| words(&n.spec) == words(&spec)) {
            Some(n) => match run(&n) {
                Ok(r) => Shown::Trained(n, Rc::new(r)),
                Err(e) => Shown::Refused(e),
            },
            None => match typed(&spec) {
                Ok(r) => Shown::Typed(r),
                Err(e) => Shown::Refused(e),
            },
        };
        Model { spec, shown, ms: now() - t, training: None, notice: None }
    }

    /// The spec's training from its starting state, playing.
    fn train(&self) -> Result<Training, String> {
        let sizes = trainable(&self.spec)?;
        let step = step_written(&self.spec, &sizes)?;
        let a = train(&self.spec, &sizes, None, 0, &[])?;
        Ok(Training { history: vec![(0, a.loss, a.right)], sizes, step: Rc::new(step), after: Rc::new(a), steps: 0, playing: true, ms_per_step: 0.0 })
    }

    /// The next CHUNK steps from the state the page holds.
    fn tick(&self, t: &Training) -> Result<Training, String> {
        let earlier: Vec<(f64, f64)> = t.history.iter().map(|h| (h.1, h.2)).collect();
        let start = now();
        let a = train(&self.spec, &t.sizes, Some(&t.after.state), CHUNK, &earlier)?;
        let steps = t.steps + CHUNK;
        let mut history = t.history.clone();
        history.push((steps, a.loss, a.right));
        Ok(Training { ms_per_step: (now() - start) / CHUNK as f64, steps, history, playing: t.playing && steps < LIMIT, after: Rc::new(a), ..t.clone() })
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl Reducible for Model {
    type Action = Action;

    fn reduce(self: Rc<Self>, action: Action) -> Rc<Self> {
        let m = (*self).clone();
        let stopped = |e: String| Model { training: m.training.clone().map(|t| Training { playing: false, ..t }), notice: Some(format!("X_eTaL stopped: {e}")), ..m.clone() };
        Rc::new(match (action, &m.training) {
            (Action::Spec(s), _) => Model::of(s),
            (Action::Train, _) => match m.train() {
                Ok(t) => Model { training: Some(t), notice: None, ..m.clone() },
                Err(e) => stopped(e),
            },
            (Action::Tick, Some(t)) if t.steps < LIMIT => match m.tick(t) {
                Ok(t) => Model { training: Some(t), notice: None, ..m.clone() },
                Err(e) => stopped(e),
            },
            (Action::Play(p), Some(t)) => Model { training: Some(Training { playing: p && t.steps < LIMIT, ..t.clone() }), ..m.clone() },
            _ => Model { training: m.training.clone().map(|t| Training { playing: false, ..t }), ..m.clone() },
        })
    }
}
