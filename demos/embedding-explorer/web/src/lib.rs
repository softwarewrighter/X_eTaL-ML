//! An embedding explorer: the demo's own X_eTaL program
//! (embedding-explorer.xtl) run once, giving each word's place in 3-D,
//! its cluster and its nearest words; the page turns the cloud and
//! shows a word's neighbors. `micro` is the model (tested natively);
//! the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
