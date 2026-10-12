//! A tiny GPT in the browser: the demo's own X_eTaL program
//! (micro-gpt.xtl) run for each set of names: the model's scores, the
//! names drawn by uniform numbers the page supplies, and for one name
//! its next-letter probabilities and its heads' attention. `micro` is
//! the model (tested natively); the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
