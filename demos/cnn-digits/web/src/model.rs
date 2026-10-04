//! The page's state and what each control does to it. The program runs
//! when the digit changes: a sample picked, a stroke finished, the pad
//! cleared.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, Anatomy, Input};
use crate::view::Stage;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub input: Input,
    /// The drawing pad, 28 x 28, 0 to 1.
    pub pad: Vec<f64>,
    pub anatomy: Option<Rc<Anatomy>>,
    /// The convolution output inspected: filter, row, column.
    pub sel: (usize, usize, usize),
    pub focus: Stage,
    /// Milliseconds the last run took.
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Sample(usize),
    /// Ink from one (row, column) of the pad to another (the same for a
    /// dot), a soft round brush along the line.
    Paint((usize, usize), (usize, usize)),
    /// A stroke finished: run the drawn digit.
    Done,
    Clear,
    Select(usize, usize, usize),
    Focus(Stage),
}

impl Model {
    pub fn new() -> Self {
        Model { input: Input::Sample(7), pad: vec![0.0; 784], anatomy: None, sel: (0, 12, 12), focus: Stage::Conv, ms: 0.0, notice: None }
            .rerun(Input::Sample(7))
    }

    /// Run `input`; on an X_eTaL error keep the last good state.
    fn rerun(self, input: Input) -> Self {
        let t = now();
        match run(&input) {
            Ok(a) => {
                // The inspector follows the filter's strongest answer.
                let f = self.sel.0;
                let (r, c) = a.strongest(f);
                Model { pad: a.x.clone(), anatomy: Some(Rc::new(a)), sel: (f, r, c), ms: now() - t, notice: None, input, ..self }
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

/// Ink a soft round brush into the pad at (r, c).
pub fn paint(pad: &mut [f64], r: usize, c: usize) {
    for (dr, dc, v) in [(0, 0, 1.0), (-1, 0, 0.6), (1, 0, 0.6), (0, -1, 0.6), (0, 1, 0.6), (-1, -1, 0.25), (-1, 1, 0.25), (1, -1, 0.25), (1, 1, 0.25)] {
        let (rr, cc) = (r as i64 + dr, c as i64 + dc);
        if (0..28).contains(&rr) && (0..28).contains(&cc) {
            let i = rr as usize * 28 + cc as usize;
            pad[i] = (pad[i] + v).min(1.0);
        }
    }
}

/// Ink the line from a to b (row, column), a brush at every cell along
/// it, so a fast stroke stays unbroken.
pub fn stroke(pad: &mut [f64], a: (usize, usize), b: (usize, usize)) {
    let (dr, dc) = (b.0 as f64 - a.0 as f64, b.1 as f64 - a.1 as f64);
    let n = dr.abs().max(dc.abs()).ceil() as usize;
    for i in 0..=n {
        let t = if n == 0 { 0.0 } else { i as f64 / n as f64 };
        let (r, c) = ((a.0 as f64 + t * dr).round() as usize, (a.1 as f64 + t * dc).round() as usize);
        paint(pad, r.min(27), c.min(27));
    }
}

impl Reducible for Model {
    type Action = Action;

    fn reduce(self: Rc<Self>, action: Action) -> Rc<Self> {
        let m = (*self).clone();
        Rc::new(match action {
            Action::Sample(d) => m.rerun(Input::Sample(d.min(9))),
            Action::Paint(a, b) => {
                let mut pad = m.pad.clone();
                stroke(&mut pad, a, b);
                Model { pad, ..m }
            }
            Action::Done => {
                let pad = m.pad.clone();
                m.rerun(Input::Drawn(pad))
            }
            Action::Clear => Model { pad: vec![0.0; 784], ..m },
            Action::Select(f, r, c) => Model { sel: (f.min(7), r.min(25), c.min(25)), ..m },
            Action::Focus(f) => Model { focus: f, ..m },
        })
    }
}
