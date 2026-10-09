//! The page's state: the training state (None before the first run),
//! the steps taken, the history the curves draw, playing or not.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, After};

/// Steps per run (a frame).
pub const CHUNK: usize = 2;
/// Where training stops by itself: 60 steps, 1,200 digits (each
/// training digit seen twice).
pub const LIMIT: usize = 60;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub after: Option<Rc<After>>,
    pub steps: usize,
    /// (steps, test loss, test share right) after each run.
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
}

impl Model {
    pub fn new() -> Self {
        let m = Model { after: None, steps: 0, history: vec![], playing: false, ms_per_step: 0.0, notice: None };
        match run(None, 0, &[]) {
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
                let earlier: Vec<(f64, f64)> = m.history.iter().map(|h| (h.1, h.2)).collect();
                let state = m.after.as_ref().map(|a| a.state.clone());
                let t = now();
                match run(state.as_deref().filter(|_| m.steps > 0), CHUNK, &earlier) {
                    Ok(a) => {
                        let steps = m.steps + CHUNK;
                        let mut history = m.history.clone();
                        history.push((steps, a.loss, a.right));
                        Model { ms_per_step: (now() - t) / CHUNK as f64, steps, history, playing: m.playing && steps < LIMIT, after: Some(Rc::new(a)), notice: None }
                    }
                    Err(e) => Model { playing: false, notice: Some(format!("X_eTaL stopped: {e}")), ..m },
                }
            }
            Action::Play(p) => Model { playing: p && m.steps < LIMIT, ..m },
            Action::Reset => Model::new(),
        })
    }
}
