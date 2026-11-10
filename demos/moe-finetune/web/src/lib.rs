//! MoE fine-tuning in the browser: the demo's own X_eTaL program
//! (moe-finetune.xtl) fine-tunes the base by each method in turn, one
//! run at a time, and the page gathers what each run measured.
//! `micro` is the model (tested natively); `app` is the page.

pub mod app;
pub mod micro;
