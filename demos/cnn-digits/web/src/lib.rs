//! The tiny CNN in the browser: the demo's own X_eTaL program
//! (cnn-digits.xtl, its data from data/) run on a sample digit or one
//! drawn on the page, every stage read back and shown. `micro` is the
//! model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
