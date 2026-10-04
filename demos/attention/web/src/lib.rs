//! The attention microscope in the browser: the demo's own X_eTaL
//! program (attention.xtl, its vocabulary and head from data/) run on
//! the sentence the page is given, every array read back and shown.
//! `micro` is the model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
