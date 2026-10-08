//! The page's state: the weights (the starting ones until a step is
//! taken), the learning rate, the step X_eTaL computed from them, and
//! the losses so far.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, Setup, Step};
use crate::view::Stage;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub setup: Setup,
    pub step: Option<Rc<Step>>,
    /// The loss at each step taken, the current one last.
    pub losses: Vec<f64>,
    pub focus: Stage,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    /// Take the step shown: its new weights become the current ones.
    Take,
    Reset,
    Rate(f64),
    Focus(Stage),
}

impl Model {
    pub fn new() -> Self {
        Model { setup: Setup::default(), step: None, losses: vec![], focus: Stage::Backward, ms: 0.0, notice: None }.rerun(Setup::default())
    }

    fn rerun(self, setup: Setup) -> Self {
        let t = now();
        match run(&setup) {
            Ok(s) => {
                let losses = [setup.earlier.clone(), vec![s.loss.0]].concat();
                Model { setup, step: Some(Rc::new(s)), losses, ms: now() - t, notice: None, ..self }
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
            Action::Take => match m.step.clone() {
                Some(s) => {
                    let setup = Setup { weights: Some((s.v1.clone(), s.v2.clone())), earlier: m.losses.clone(), ..m.setup.clone() };
                    m.rerun(setup)
                }
                None => m,
            },
            Action::Reset => Model { losses: vec![], ..m }.rerun(Setup::default()),
            Action::Rate(lr) => {
                let setup = Setup { lr, ..m.setup.clone() };
                m.rerun(setup)
            }
            Action::Focus(f) => Model { focus: f, ..m },
        })
    }
}
