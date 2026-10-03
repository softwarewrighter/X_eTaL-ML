//! The MoE routing microscope in the browser: the demo's own X_eTaL
//! program (scores, softmax, top-2, load) run on a sentence the page
//! picks. `micro` is the model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
