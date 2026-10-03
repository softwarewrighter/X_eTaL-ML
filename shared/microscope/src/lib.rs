//! The shared shell of the X_eTaL ML demos' web pages (the "microscope"):
//! every page runs an X_eTaL program, reads back the arrays it prints,
//! and shows them beside the program's decorated source.
//!
//! - `run`: run a program, read printed arrays, write literals
//! - `source`: decorated source with a highlighted range; shapes
//! - `canvas`, `colour`: large arrays as pixels
//! - `cells`: small boards as clickable cells
//! - `chrome`: header, stage chips, panels, notices, footer
//!
//! Pages also link `microscope.css` (trunk: `rel="css"`).

pub mod canvas;
pub mod cells;
pub mod chrome;
pub mod colour;
pub mod run;
pub mod source;
