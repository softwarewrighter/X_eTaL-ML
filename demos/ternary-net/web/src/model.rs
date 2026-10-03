//! The page's state and what each control does to it. Three programs
//! (all from ternary-net.xtl) run only when what they depend on
//! changes: the formats on the weight set, the ternary weights on the
//! threshold, the point on any of them.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run_formats, run_point, run_ternary, Formats, Point, Setup, Ternary};
use crate::view::Stage;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub setup: Setup,
    pub formats: Option<Rc<Formats>>,
    pub ternary: Option<Rc<Ternary>>,
    pub point: Option<Rc<Point>>,
    /// The layer-2 output inspected.
    pub unit: usize,
    pub focus: Stage,
    /// Milliseconds the last run of each program took.
    pub ms: (f64, f64, f64),
    pub notice: Option<String>,
}

pub enum Action {
    Set(usize),
    Thresh(f64),
    Point(f64, f64),
    Unit(usize),
    Focus(Stage),
}

fn timed<T>(f: impl FnOnce() -> Result<T, String>) -> (Result<T, String>, f64) {
    let t = now();
    let r = f();
    (r, now() - t)
}

impl Model {
    pub fn new() -> Self {
        Model { setup: Setup::default(), formats: None, ternary: None, point: None, unit: 0, focus: Stage::Additions, ms: (0.0, 0.0, 0.0), notice: None }
            .rerun(Setup::default(), 3)
    }

    /// Run the last `n` programs (3: all, 2: ternary and point, 1:
    /// point) for `setup`; on an X_eTaL error keep the last good state.
    fn rerun(self, setup: Setup, n: usize) -> Self {
        let mut m = Model { notice: None, ..self };
        let fail = |m: Model, e: String| Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last good run)")), ..m };
        if n >= 3 {
            match timed(|| run_formats(&setup)) {
                (Ok(f), ms) => {
                    m.formats = Some(Rc::new(f));
                    m.ms.0 = ms;
                }
                (Err(e), _) => return fail(m, e),
            }
        }
        if n >= 2 {
            let y32 = m.formats.as_ref().map(|f| f.y32.clone()).unwrap_or_default();
            match timed(|| run_ternary(&setup, &y32)) {
                (Ok(t), ms) => {
                    m.ternary = Some(Rc::new(t));
                    m.ms.1 = ms;
                }
                (Err(e), _) => return fail(m, e),
            }
        }
        match timed(|| run_point(&setup)) {
            (Ok(p), ms) => {
                m.point = Some(Rc::new(p));
                m.ms.2 = ms;
            }
            (Err(e), _) => return fail(m, e),
        }
        Model { setup, ..m }
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
        let s = m.setup.clone();
        Rc::new(match action {
            Action::Set(i) => m.rerun(Setup { set: i, ..s }, 3),
            Action::Thresh(t) => m.rerun(Setup { t, ..s }, 2),
            Action::Point(x, y) => m.rerun(Setup { point: (x, y), ..s }, 1),
            Action::Unit(j) => Model { unit: j.min(15), ..m },
            Action::Focus(f) => Model { focus: f, ..m },
        })
    }
}
