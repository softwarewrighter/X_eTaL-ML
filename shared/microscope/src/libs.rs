//! This repo's libraries (libs/<Name>/src/<Name>.xtl, embedded at build
//! time by build.rs) in an in-memory store, so a page's program finds
//! them with `u_se<` as the command line does through XETAL_PATH.
//! `run::output` installs it before the first run. A page adds its
//! demo's data files with `add`, as `n_umbers []N_GET "data/..."` reads
//! them at the command line.
//!
//! Another repository's pages get their own libraries in two ways
//! (X_eTaL-fine-tuning's ask FM1): name their roots in MICROSCOPE_LIBS
//! when building (build.rs embeds them after this repository's), or
//! put each library's text in the store with `add("Name.xtl", source)`
//! before the first run; `u_se< "Name"` then finds it.

use std::sync::{Arc, OnceLock};

use xetal_store::{Memory, Store};

include!(concat!(env!("OUT_DIR"), "/libs.rs"));

/// The one store: every library, then the files pages add.
fn store() -> &'static Arc<Memory> {
    static STORE: OnceLock<Arc<Memory>> = OnceLock::new();
    STORE.get_or_init(|| {
        let store = Arc::new(Memory::default());
        for (file, source) in LIBRARIES {
            let _ = store.put(file, source);
        }
        xetal_store::install(store.clone());
        store
    })
}

/// Install the store holding every library (once; later calls do nothing).
pub fn install() {
    store();
}

/// The libraries embedded, by file name (`LoRA.xtl`, `Flow.xtlm`).
pub fn names() -> Vec<&'static str> {
    LIBRARIES.iter().map(|(file, _)| *file).collect()
}

/// Put a file in the store: a demo's data (`data/weights.txt`, where the
/// program's `[]N_GET` finds it) or a library (`Name.xtl` or
/// `Name.xtlm`, which `u_se< "Name"` then finds).
pub fn add(path: &str, text: &str) {
    let _ = store().put(path, text);
}
