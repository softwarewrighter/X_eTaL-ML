//! This repo's libraries (libs/<Name>/src/<Name>.xtl, embedded at build
//! time by build.rs) in an in-memory store, so a page's program finds
//! them with `u_se<` as the command line does through XETAL_PATH.
//! `run::output` installs it before the first run.

use std::sync::{Arc, Once};

use xetal_store::{Memory, Store};

include!(concat!(env!("OUT_DIR"), "/libs.rs"));

/// Install the store holding every library (once; later calls do nothing).
pub fn install() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let store = Memory::default();
        for (file, source) in LIBRARIES {
            let _ = store.put(file, source);
        }
        xetal_store::install(Arc::new(store));
    });
}
