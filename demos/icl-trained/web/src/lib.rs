//! In-context learning, trained, in the browser: the demo's own X_eTaL
//! program (icl-trained.xtl) runs a small transformer trained on Gao and
//! Gao's task over a prompt, with an edit or none, and takes it apart.
//! `micro` is the model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
