//! Training a CNN in the browser: the demo's own X_eTaL program
//! (cnn-backprop.xtl) run two Adam steps at a time from the state the
//! page holds, the filters, the test digits' readings and the curves
//! redrawn after each run. `micro` is the model (tested natively); the
//! rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
