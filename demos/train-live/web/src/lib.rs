//! Training live in the browser: the demo's own X_eTaL program
//! (train-live.xtl) run a few dozen Adam steps at a time from the state
//! the page holds, the decision map and the curves redrawn after each.
//! `micro` is the model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
