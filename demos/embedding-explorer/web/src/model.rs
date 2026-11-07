//! The page's state: the cloud X_eTaL computed, how it is turned, the
//! word picked, spinning or not.

use std::rc::Rc;

use microscope::run::now;
use microscope::turning::{key, turn, Angle};
use yew::Reducible;

use crate::micro::{run, words, Cloud};

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub cloud: Option<Rc<Cloud>>,
    /// Turned about the vertical axis, then tipped, in radians.
    pub yaw: f64,
    pub pitch: f64,
    pub spinning: bool,
    /// The word picked (from 0).
    pub picked: usize,
    /// Where a drag last was.
    pub drag: Option<(f64, f64)>,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Down(f64, f64),
    Move(f64, f64),
    Up,
    Spin(bool),
    Tick,
    Pick(usize),
    /// An arrow key, once the cloud has the focus.
    Key(String),
}

impl Model {
    pub fn new() -> Self {
        let t = now();
        let picked = words().iter().position(|w| *w == "cat").unwrap_or(0);
        let m = Model { cloud: None, yaw: 0.6, pitch: 0.35, spinning: true, picked, drag: None, ms: 0.0, notice: None };
        match run() {
            Ok(c) => Model { cloud: Some(Rc::new(c)), ms: now() - t, ..m },
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
            Action::Down(x, y) => Model { drag: Some((x, y)), spinning: false, ..m },
            Action::Move(x, y) => match m.drag {
                Some((x0, y0)) => {
                    let a = turn(Angle { yaw: m.yaw, pitch: m.pitch }, x - x0, y - y0);
                    Model { yaw: a.yaw, pitch: a.pitch, drag: Some((x, y)), ..m }
                }
                None => m,
            },
            Action::Key(k) => match key(Angle { yaw: m.yaw, pitch: m.pitch }, &k) {
                Some(a) => Model { yaw: a.yaw, pitch: a.pitch, spinning: false, ..m },
                None => m,
            },
            Action::Up => Model { drag: None, ..m },
            Action::Spin(s) => Model { spinning: s, ..m },
            Action::Tick => match m.spinning {
                true => Model { yaw: m.yaw + 0.012, ..m },
                false => m,
            },
            Action::Pick(i) => Model { picked: i, ..m },
        })
    }
}
