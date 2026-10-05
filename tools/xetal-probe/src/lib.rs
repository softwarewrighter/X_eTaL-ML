//! The part of the pinned `xetal-play` API the demos rely on.

/// Run a program, returning (output, errors).
pub fn run(src: &str) -> (String, String) {
    let r = xetal_play::run(src, 1);
    (r.out, r.err)
}
