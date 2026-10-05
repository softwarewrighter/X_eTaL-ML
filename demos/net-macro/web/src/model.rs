//! The page's state: the spec in the box, and what the macro made of
//! it. A spec that is one of the program's networks runs that network;
//! any other is expanded and counted.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{clean, networks, run, typed, Network, Trained, Typed};

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
}

pub enum Action {
    Spec(String),
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
        Model { spec, shown, ms: now() - t }
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
        let Action::Spec(s) = action;
        Rc::new(Model::of(s))
    }
}
