//! The page's state: the temperature, the starting letters, the uniform
//! numbers of this set of names, the name looked into, what X_eTaL made.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{clean, run, uniforms, Names};

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub t: f64,
    pub start: String,
    /// Which set of uniform numbers (the browser draws fresh ones each time).
    pub set: u64,
    pub u: Rc<Vec<f64>>,
    /// The name looked into, from 1.
    pub pick: usize,
    pub names: Option<Rc<Names>>,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Invent,
    Temp(f64),
    Start(String),
    Pick(usize),
}

impl Model {
    pub fn new() -> Self {
        Model { t: 0.5, start: String::new(), set: 1, u: Rc::new(uniforms(1)), pick: 1, names: None, ms: 0.0, notice: None }.rerun()
    }

    fn rerun(self) -> Self {
        let t = now();
        match run(self.t, &self.start, &self.u, self.pick) {
            Ok(r) => Model { names: Some(Rc::new(r)), ms: now() - t, notice: None, ..self },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e}")), ..self },
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
            Action::Invent => Model { set: m.set + 1, u: Rc::new(uniforms(m.set + 1)), pick: 1, ..m }.rerun(),
            Action::Temp(t) => Model { t, pick: 1, ..m }.rerun(),
            Action::Start(s) => Model { start: clean(&s), pick: 1, ..m }.rerun(),
            Action::Pick(i) => Model { pick: i, ..m }.rerun(),
        })
    }
}
