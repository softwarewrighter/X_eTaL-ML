//! The page's state and what each control does to it. A run is queued
//! and done after the page has drawn "running", so a click answers at
//! once.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{paper, presets, run, Anatomy, Config, Prompt};

#[derive(Clone, Debug, PartialEq)]
pub enum Job {
    Run,
    Paper,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    /// The prompt as typed.
    pub text: String,
    pub prompt: Prompt,
    pub config: Config,
    pub anatomy: Option<Rc<Anatomy>>,
    /// The paper's knockouts, once run.
    pub paper: Option<Rc<Vec<(f64, f64)>>>,
    pub queue: Vec<Job>,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Text(String),
    Preset(usize),
    Config(Config),
    Paper,
    Next,
}

impl Model {
    pub fn new() -> Self {
        let p = presets()[0].1.clone();
        Model { text: p.text(), prompt: p, config: Config::default(), anatomy: None, paper: None, queue: vec![Job::Run], ms: 0.0, notice: None }
    }

    fn queued(self, job: Job) -> Self {
        let mut queue = self.queue.clone();
        if !queue.contains(&job) {
            queue.push(job);
        }
        Model { queue, ..self }
    }

    /// Do the first job; on an X_eTaL error keep the last good state.
    fn next(self) -> Self {
        let Some(job) = self.queue.first().cloned() else { return self };
        let queue = self.queue[1..].to_vec();
        let t = now();
        match job {
            Job::Run => match run(&self.prompt, &self.config) {
                Ok(a) => Model { anatomy: Some(Rc::new(a)), ms: now() - t, notice: None, queue, ..self },
                Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last good run)")), queue, ..self },
            },
            Job::Paper => match paper() {
                Ok(v) => Model { paper: Some(Rc::new(v)), queue, ..self },
                Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e}")), queue, ..self },
            },
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
            Action::Text(s) => match Prompt::parse(&s) {
                Ok(p) => Model { text: s, prompt: p, notice: None, ..m }.queued(Job::Run),
                Err(e) => Model { text: s, notice: Some(e), ..m },
            },
            Action::Preset(i) => {
                let p = presets()[i].1.clone();
                Model { text: p.text(), prompt: p, notice: None, ..m }.queued(Job::Run)
            }
            Action::Config(c) => Model { config: c, ..m }.queued(Job::Run),
            Action::Paper => m.queued(Job::Paper),
            Action::Next => m.next(),
        })
    }
}
