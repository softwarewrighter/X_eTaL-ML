//! The teaching shell of X_eTaL pages, on top of `microscope` (which
//! runs X_eTaL, reads arrays back and draws decorated source). Written
//! in X_eTaL-fine-tuning for its lessons and shared from X_eTaL-ML
//! (X_eTaL-manifolds' ask GM1): a dependent names its site, root,
//! glossary files and pins at build time (build.rs).
//!
//! - `chrome`: the site's header and footer, the frames of a page,
//!   asides
//! - `code`: X_eTaL drawn with every function linked to the cross-reference
//! - `eli5`: every idea explained simply, opened from the word itself
//! - `explain`: a library function explained from its own source
//!
//! - `mode`: three depths over one computation, and their switch
//! - `heat`: a matrix as a heat map with a legend and a cell inspector
//! - `stepper`: a pipeline stepped through, both ways, from its trace
//! - `arithmetic`: a sum written out in symbols and with its numbers
//! - `narrate`: what changed, in words, from measured numbers
//! - `predict`: a question answered before the experiment is shown
//! - `link`: a page opened at a state named in its address
//!
//! Written for a reader with new eyes: one who knows neither machine
//! learning nor X_eTaL (docs/new-eyes.md). Presentation only: nothing
//! here computes a number a page shows.

pub mod arithmetic;
pub mod chrome;
pub mod code;
pub mod eli5;
pub mod explain;
pub mod heat;
pub mod link;
pub mod mode;
pub mod narrate;
pub mod predict;
pub mod stepper;
