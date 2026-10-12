//! The page's state: the start state, the window's three positions,
//! the dreamed ball's path and the true ball's, the last prediction,
//! playing or not.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{begin, exact, held_out, step, Place, State, Step, SPEEDS, TOP};

/// Where dreaming stops by itself.
pub const LIMIT: usize = 40;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The state the window starts from.
    pub start: State,
    /// The three positions the network sees next (the dream's last three).
    pub window: [Place; 3],
    /// The true state of the frame the next prediction is of.
    pub truth: State,
    /// The dreamed ball after each step, and the true ball.
    pub dreamed: Vec<Place>,
    pub real: Vec<Place>,
    /// Whether each step's predicted frame was exactly the true one.
    pub exact: Vec<bool>,
    pub last: Option<Rc<Step>>,
    pub steps: usize,
    pub playing: bool,
    pub ms: f64,
    pub notice: Option<String>,
    /// Which held-out state the Unseen button gives next.
    pub unseen: usize,
}

pub enum Action {
    Tick,
    Play(bool),
    Reset,
    /// The ball put at x y (its top left).
    Put(i64, i64),
    Vx(i64),
    Vy(i64),
    /// The next state the network never trained on.
    Unseen,
}

impl Model {
    pub fn new() -> Self {
        let m = Self::begin(held_out()[0], 1);
        Model { ..m }
    }

    /// The window from the state s, not yet stepped.
    fn begin(s: State, unseen: usize) -> Self {
        let m = Model {
            start: s,
            window: [(0, 0); 3],
            truth: s,
            dreamed: vec![],
            real: vec![],
            exact: vec![],
            last: None,
            steps: 0,
            playing: false,
            ms: 0.0,
            notice: None,
            unseen,
        };
        let t = now();
        match begin(s) {
            Ok(b) => Model { window: b.window, truth: b.truth, ms: now() - t, ..m },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e}")), ..m },
        }
    }

    /// Whether the start is one the network never trained on.
    pub fn is_unseen(&self) -> bool {
        held_out().contains(&self.start)
    }

    /// Steps so far where the dreamed ball was where the true one is.
    pub fn on_track(&self) -> usize {
        self.dreamed.iter().zip(&self.real).filter(|(d, r)| d == r).count()
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
        let s = m.start;
        Rc::new(match action {
            Action::Tick => {
                if m.steps >= LIMIT {
                    return Rc::new(Model { playing: false, ..m });
                }
                let t = now();
                match step(&m.window, m.truth) {
                    Ok(st) => {
                        let (mut dreamed, mut real, mut exact_) = (m.dreamed.clone(), m.real.clone(), m.exact.clone());
                        dreamed.push(st.place);
                        real.push((m.truth[0], m.truth[1]));
                        exact_.push(exact(&st.probs, m.truth));
                        let steps = m.steps + 1;
                        let window = [m.window[1], m.window[2], st.place];
                        let playing = m.playing && steps < LIMIT;
                        Model { window, truth: st.next, dreamed, real, exact: exact_, steps, playing, ms: now() - t, last: Some(Rc::new(st)), notice: None, ..m }
                    }
                    Err(e) => Model { playing: false, notice: Some(format!("X_eTaL stopped: {e}")), ..m },
                }
            }
            Action::Play(p) => Model { playing: p && m.steps < LIMIT, ..m },
            Action::Reset => Model::begin(s, m.unseen),
            Action::Put(x, y) => Model::begin([x.clamp(0, TOP), y.clamp(0, TOP), s[2], s[3]], m.unseen),
            Action::Vx(v) if SPEEDS.contains(&v) => Model::begin([s[0], s[1], v, s[3]], m.unseen),
            Action::Vy(v) if SPEEDS.contains(&v) => Model::begin([s[0], s[1], s[2], v], m.unseen),
            Action::Vx(_) | Action::Vy(_) => m,
            Action::Unseen => {
                let held = held_out();
                Model::begin(held[m.unseen % held.len()], m.unseen + 1)
            }
        })
    }
}
