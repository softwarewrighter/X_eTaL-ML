//! This repo's libraries (libs/<Name>/src/<Name>.xtl, embedded at build
//! time by build.rs) in an in-memory store, so a page's program finds
//! them with `u_se<` as the command line does through XETAL_PATH.
//! `run::output` installs it before the first run. A page adds its
//! demo's data files with `add`, as `n_umbers []N_GET "data/..."` reads
//! them at the command line.

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

/// Put a file (a demo's data, `data/weights.txt`) in the store, where
/// the program's `[]N_GET` finds it.
pub fn add(path: &str, text: &str) {
    let _ = store().put(path, text);
}
