//! The page's state and what each control does to it. The program runs
//! when the sentence or the mask changes.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, Anatomy};
use crate::view::Stage;

pub const PRESETS: [&str; 3] = [
    "the animal did not cross the street because it was tired",
    "the animal did not cross the street because it was wide",
    "the dog was hungry because the road was busy",
];

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub sentence: String,
    pub causal: bool,
    pub anatomy: Option<Rc<Anatomy>>,
    /// The token inspected.
    pub row: usize,
    pub focus: Stage,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Sentence(String),
    Causal(bool),
    Row(usize),
    Focus(Stage),
}

impl Model {
    pub fn new() -> Self {
        Model { sentence: PRESETS[0].into(), causal: false, anatomy: None, row: 10, focus: Stage::Weights, ms: 0.0, notice: None }.rerun()
    }

    /// Run the sentence; on an X_eTaL error keep the last good state.
    fn rerun(self) -> Self {
        let t = now();
        match run(&self.sentence, self.causal) {
            Ok(a) => {
                let row = self.row.min(a.n() - 1);
                Model { anatomy: Some(Rc::new(a)), row, ms: now() - t, notice: None, ..self }
            }
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last good run)")), ..self },
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
            Action::Sentence(s) => Model { sentence: s, ..m }.rerun(),
            Action::Causal(c) => Model { causal: c, ..m }.rerun(),
            Action::Row(i) => Model { row: i, ..m },
            Action::Focus(f) => Model { focus: f, ..m },
        })
    }
}
