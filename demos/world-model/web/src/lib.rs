//! A tiny world model in the browser: the demo's own X_eTaL program
//! (world-model.xtl) run one step at a time from the three ball
//! positions and the true state the page holds. `micro` is the model
//! (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
