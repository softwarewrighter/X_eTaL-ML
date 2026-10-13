//! A learned denoiser in the browser: the demo's own X_eTaL program
//! (diffusion.xtl) run on eight pictures of noise the page supplies,
//! every level of the way kept. `micro` is the model (tested natively);
//! the rest is the page.

pub mod app;
pub mod micro;
pub mod model;
pub mod view;
