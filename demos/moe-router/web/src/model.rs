//! The page's state and what each control does to it.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{run, run_nudge, tokens, Anatomy, Nudge, Nudged, SENTENCES};
use crate::view::Stage;

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub sentence: String,
    pub last: Option<Rc<Anatomy>>,
    /// The token inspected (and highlighted in the picture).
    pub token: usize,
    pub playing: bool,
    pub focus: Stage,
    pub ms: f64,
    pub notice: Option<String>,
    /// The nudge: what is pushed where, the result, the sample shown.
    pub nudge: Nudge,
    pub nudged: Option<Rc<Nudged>>,
    pub cursor: usize,
    pub nudge_ms: f64,
}

pub enum Action {
    Sentence(String),
    Token(usize),
    Tick,
    TogglePlay,
    Focus(Stage),
    Nudge(Nudge),
    Cursor(usize),
}

impl Model {
    pub fn new() -> Self {
        let m = Model {
            sentence: String::new(),
            last: None,
            token: 0,
            playing: true,
            focus: Stage::Top2,
            ms: 0.0,
            notice: None,
            nudge: Nudge::default(),
            nudged: None,
            cursor: 60,
            nudge_ms: 0.0,
        };
        m.route(SENTENCES[0].to_string()).push(Nudge::default())
    }

    /// Route `sentence`; on an error keep the last good routing.
    fn route(self, sentence: String) -> Self {
        if tokens(&sentence).len() < 2 {
            return Model { notice: Some("Type a sentence of at least two words (showing the last sentence).".into()), ..self };
        }
        let t = now();
        match run(&sentence) {
            Ok(a) => Model { sentence, last: Some(Rc::new(a)), token: 0, ms: now() - t, notice: None, ..self },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last sentence)")), ..self },
        }
    }

    /// Run the nudge; on an error keep the last good one.
    fn push(self, nudge: Nudge) -> Self {
        let t = now();
        match run_nudge(&nudge) {
            Ok(n) => Model { nudge, nudged: Some(Rc::new(n)), nudge_ms: now() - t, ..self },
            Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last nudge)")), ..self },
        }
    }

    fn tokens(&self) -> usize {
        self.last.as_ref().map_or(1, |a| a.n()).max(1)
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
            Action::Sentence(s) => m.route(s),
            Action::Token(t) => Model { token: t.min(m.tokens() - 1), playing: false, ..m },
            Action::Tick => Model { token: (m.token + 1) % m.tokens(), ..m },
            Action::TogglePlay => Model { playing: !m.playing, ..m },
            Action::Focus(f) => Model { focus: f, ..m },
            Action::Nudge(n) => m.push(n),
            Action::Cursor(i) => Model { cursor: i.min(crate::micro::STEPS - 1), ..m },
        })
    }
}
