//! The backprop microscope in the browser: the demo's own X_eTaL
//! program (backprop.xtl, its batch and starting weights from data/)
//! run on the current weights, every array of one training step read
//! back and shown. `micro` is the model (tested natively); the rest is
//! the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
