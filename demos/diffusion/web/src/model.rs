//! The page's state: the noise the pictures start from, their journey
//! down the levels, and which picture is looked at.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, seeded, Journey, PICTURES};

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub noise: Vec<f64>,
    pub journey: Option<Rc<Journey>>,
    /// The picture looked at, 0 to PICTURES - 1.
    pub pick: usize,
    /// Runs so far (the first is from a fixed seed, the rest from the browser's randomness).
    pub runs: usize,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    /// New pictures of noise (PICTURES x N Gaussian numbers).
    Draw(Vec<f64>),
    /// The same noise again.
    Again,
    Pick(usize),
}

impl Model {
    pub fn new() -> Self {
        Self::from_noise(seeded(2026), 0, 0)
    }

    fn from_noise(noise: Vec<f64>, pick: usize, runs: usize) -> Self {
        let t = now();
        let m = Model { noise, journey: None, pick, runs: runs + 1, ms: 0.0, notice: None };
        match run(&m.noise) {
            Ok(j) => Model { journey: Some(Rc::new(j)), ms: now() - t, ..m },
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
            Action::Draw(noise) => Model::from_noise(noise, m.pick, m.runs),
            Action::Again => Model::from_noise(m.noise.clone(), m.pick, m.runs),
            Action::Pick(i) => Model { pick: i.min(PICTURES - 1), ..m },
        })
    }
}
