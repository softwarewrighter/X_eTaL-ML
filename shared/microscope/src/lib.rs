//! The shared shell of the X_eTaL ML demos' web pages (the "microscope"):
//! every page runs an X_eTaL program, reads back the arrays it prints,
//! and shows them beside the program's decorated source.
//!
//! - `run`: run a program, read printed arrays, write literals
//! - `source`: decorated source with a highlighted range; shapes
//! - `canvas`, `color`: large arrays as pixels
//! - `cells`: small boards as clickable cells
//! - `heat`: small matrices as heat maps, colored as the 3D stage view
//! - `chrome`: header, stage chips, panels, pictures, notices, footer
//! - `libs`: this repo's libraries, importable with `u_se<`
//! - `palette`: composing an equation or a pipeline from parts, run live
//! - `stages`, `stageview`: a pipeline's stages, drawn as math and in 3D
//! - `turning`: points, segments and wire surfaces in 3D, turned by
//!   pointer, keys and touch
//!
//! Pages also link `microscope.css` (trunk: `rel="css"`).

pub mod canvas;
pub mod cells;
pub mod chrome;
pub mod color;
pub mod heat;
pub mod libs;
pub mod palette;
pub mod run;
pub mod source;
pub mod stages;
pub mod stageview;
pub mod turning;
