//! The page's state and what each control does to it. A run is queued
//! and done after the page has drawn "running", so a click answers at
//! once.

use std::rc::Rc;

use microscope::run::now;
use yew::Reducible;

use crate::micro::{gate, run, Anatomy, Edit, Gate, Set};

#[derive(Clone, Debug, PartialEq)]
pub enum Job {
    Run,
    Gate,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub model: usize,
    pub set: Set,
    pub k: usize,
    pub edit: Edit,
    pub anatomy: Option<Rc<Anatomy>>,
    /// The intact run of the prompt, kept to compare an edit with.
    pub intact: Option<Rc<Anatomy>>,
    /// Each model's gate, once found.
    pub gates: [Option<Rc<Gate>>; 3],
    pub queue: Vec<Job>,
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Model(usize),
    Prompt(Set, usize),
    Edit(Edit),
    Gate,
    Next,
}

impl Model {
    pub fn new() -> Self {
        Model { model: 2, set: Set::Recurring, k: 1, edit: Edit::None, anatomy: None, intact: None, gates: [None, None, None], queue: vec![Job::Run], ms: 0.0, notice: None }
    }

    fn queued(self, job: Job) -> Self {
        let mut queue = self.queue.clone();
        if !queue.contains(&job) {
            queue.push(job);
        }
        Model { queue, ..self }
    }

    /// The gate direction for a steer: the found one, after its layer.
    pub fn direction(&self) -> Option<Vec<f64>> {
        self.gates[self.model].as_ref().map(|g| g.directions[g.layer() - 1].clone())
    }

    /// Do the first job; on an X_eTaL error keep the last good state.
    fn next(self) -> Self {
        let Some(job) = self.queue.first().cloned() else { return self };
        let queue = self.queue[1..].to_vec();
        let t = now();
        match job {
            Job::Run => {
                let d = self.direction();
                match run(self.model, self.set, self.k, self.edit, d.as_deref()) {
                    Ok(a) => {
                        let a = Rc::new(a);
                        let intact = if self.edit == Edit::None { Some(a.clone()) } else { self.intact.clone() };
                        Model { anatomy: Some(a), intact, ms: now() - t, notice: None, queue, ..self }
                    }
                    Err(e) => Model { notice: Some(format!("X_eTaL stopped: {e} (showing the last good run)")), queue, ..self },
                }
            }
            Job::Gate => match gate(self.model) {
                Ok(g) => {
                    let mut gates = self.gates.clone();
                    gates[self.model] = Some(Rc::new(g));
                    Model { gates, ms: now() - t, queue, ..self }
                }
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
            // A new model or prompt starts intact; its intact run is redone.
            Action::Model(i) => Model { model: i, edit: Edit::None, intact: None, ..m }.queued(Job::Run),
            Action::Prompt(set, k) => Model { set, k, edit: Edit::None, intact: None, ..m }.queued(Job::Run),
            Action::Edit(e) => Model { edit: e, ..m }.queued(Job::Run),
            Action::Gate => m.queued(Job::Gate),
            Action::Next => m.next(),
        })
    }
}
