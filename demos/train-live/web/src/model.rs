//! The page's state: Adam's state (None before the first run), the
//! learning rate, the steps taken, and the history the curves draw.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, After};

/// Steps per run (a frame).
pub const CHUNK: usize = 25;
/// Where training stops by itself.
pub const LIMIT: usize = 1000;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub lr: f64,
    pub after: Option<Rc<After>>,
    pub steps: usize,
    /// (steps, loss, share right) after each run.
    pub history: Vec<(usize, f64, f64)>,
    pub playing: bool,
    /// Milliseconds per step, the last run's.
    pub ms_per_step: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Tick,
    Play(bool),
    Reset,
    Rate(f64),
}

impl Model {
    pub fn new() -> Self {
        Self::start(0.02)
    }

    /// The starting state, untrained (zero steps).
    fn start(lr: f64) -> Self {
        let m = Model { lr, after: None, steps: 0, history: vec![], playing: false, ms_per_step: 0.0, notice: None };
        match run(lr, None, 0) {
            Ok(a) => Model { history: vec![(0, a.loss, a.right)], after: Some(Rc::new(a)), ..m },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e}")), ..m },
        }
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
        Rc::new(match action {
            Action::Tick => {
                if m.steps >= LIMIT {
                    return Rc::new(Model { playing: false, ..m });
                }
                let state = m.after.as_ref().map(|a| a.state.clone());
                let t = now();
                match run(m.lr, state.as_deref(), CHUNK) {
                    Ok(a) => {
                        let steps = m.steps + CHUNK;
                        let mut history = m.history.clone();
                        history.push((steps, a.loss, a.right));
                        Model { ms_per_step: (now() - t) / CHUNK as f64, steps, history, playing: m.playing && steps < LIMIT, after: Some(Rc::new(a)), notice: None, ..m }
                    }
                    Err(e) => Model { playing: false, notice: Some(format!("X_eTaL stopped: {e}")), ..m },
                }
            }
            Action::Play(p) => Model { playing: p && m.steps < LIMIT, ..m },
            Action::Reset => Model::start(m.lr),
            Action::Rate(lr) => Model::start(lr),
        })
    }
}
