//! The page's state: k, the start, the centers after each step (the
//! trails), the inertia after each, playing or not.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, After, Start};

/// Where stepping stops by itself (k-means settles well before).
pub const LIMIT: usize = 30;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub k: usize,
    pub start: Start,
    pub after: Option<Rc<After>>,
    /// The centers after each step, the start first.
    pub trail: Vec<Vec<f64>>,
    pub inertias: Vec<f64>,
    pub steps: usize,
    pub playing: bool,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Tick,
    Play(bool),
    Reset,
    K(usize),
    Start(Start),
}

impl Model {
    pub fn new() -> Self {
        Self::begin(5, Start::Poor)
    }

    /// The start, not yet stepped.
    fn begin(k: usize, start: Start) -> Self {
        let m = Model { k, start, after: None, trail: vec![], inertias: vec![], steps: 0, playing: false, ms: 0.0, notice: None };
        let t = now();
        match run(k, start, None, &[]) {
            Ok(a) => Model { trail: vec![a.centers.clone()], inertias: vec![a.inertia], after: Some(Rc::new(a)), ms: now() - t, ..m },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e}")), ..m },
        }
    }

    /// Settled: no point changed center in the last step.
    pub fn settled(&self) -> bool {
        self.steps > 0 && self.after.as_ref().is_some_and(|a| a.settled)
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
                if m.steps >= LIMIT || m.settled() {
                    return Rc::new(Model { playing: false, ..m });
                }
                let Some(centers) = m.after.as_ref().map(|a| a.centers.clone()) else { return Rc::new(m) };
                let t = now();
                match run(m.k, m.start, Some(&centers), &m.inertias) {
                    Ok(a) => {
                        let (mut trail, mut inertias) = (m.trail.clone(), m.inertias.clone());
                        trail.push(a.centers.clone());
                        inertias.push(a.inertia);
                        let steps = m.steps + 1;
                        let playing = m.playing && steps < LIMIT && !a.settled;
                        Model { trail, inertias, steps, playing, ms: now() - t, after: Some(Rc::new(a)), notice: None, ..m }
                    }
                    Err(e) => Model { playing: false, notice: Some(format!("X_eTaL stopped: {e}")), ..m },
                }
            }
            Action::Play(p) => Model { playing: p && m.steps < LIMIT && !m.settled(), ..m },
            Action::Reset => Model::begin(m.k, m.start),
            Action::K(k) => Model::begin(k, m.start),
            Action::Start(s) => Model::begin(m.k, s),
        })
    }
}
