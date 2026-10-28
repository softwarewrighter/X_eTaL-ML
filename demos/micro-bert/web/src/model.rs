//! The page's state: the name with its blank, what X_eTaL made of it.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{clean, run, Fill};

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub name: String,
    pub fill: Option<Rc<Fill>>,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Name(String),
}

impl Model {
    pub fn new() -> Self {
        Model { name: "ma_y".to_string(), fill: None, ms: 0.0, notice: None }.rerun()
    }

    fn rerun(self) -> Self {
        let t = now();
        match run(&self.name) {
            Ok(f) => Model { fill: Some(Rc::new(f)), ms: now() - t, notice: None, ..self },
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
            Action::Name(s) => Model { name: clean(&s), ..m }.rerun(),
        })
    }
}
