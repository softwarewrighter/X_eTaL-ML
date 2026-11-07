//! What changed, said in words from measured numbers (docs/plan.md,
//! principle 4: explain changes, and do not oversell). A page moves a
//! control and has the values before and after from two runs; these
//! make the sentences. They report and compare; they never call a
//! result better, and where a measurement only suggests something they
//! say "may". They compare numbers (which is larger) and do no
//! arithmetic on them: a difference or a percentage a page wants to
//! show is X_eTaL's to compute. Text only, so it is tested natively.

use crate::arithmetic::{commas, show};

/// Words a narration may not use: a judgment is the reader's, and a
/// measurement on one small task supports none of them.
pub const FORBIDDEN: [&str; 6] = ["better", "best", "worse", "always", "never", "optimal"];

/// The forbidden words a text uses (a test of every narration).
pub fn forbidden(text: &str) -> Vec<&'static str> {
    let lower = text.to_lowercase();
    FORBIDDEN.iter().copied().filter(|w| lower.split(|c: char| !c.is_alphabetic()).any(|t| t == *w)).collect()
}

/// How a measured value moved: "X rose from a to b", "X fell from a to
/// b", or that it did not move.
pub fn moved(what: &str, before: f64, after: f64) -> String {
    if before == after {
        return format!("{what} stayed at {}", show(before));
    }
    let way = if after > before { "rose" } else { "fell" };
    format!("{what} {way} from {} to {}", show(before), show(after))
}

/// How a count moved, and how it now stands beside another count (the
/// numbers of the table itself): "numbers trained went from 32 to 64,
/// as many as the 64 of the table itself".
pub fn count(what: &str, before: usize, after: usize, beside: Option<(usize, &str)>) -> String {
    let head = if before == after {
        format!("{what} stayed at {}", commas(after))
    } else {
        format!("{what} went from {} to {}", commas(before), commas(after))
    };
    let Some((other, name)) = beside else { return head };
    let tail = if after == other {
        format!("as many as the {} of {name}", commas(other))
    } else if after > other {
        format!("more than the {} of {name}", commas(other))
    } else {
        format!("fewer than the {} of {name}", commas(other))
    };
    format!("{head}, {tail}")
}

/// Two losses after a change: the one on the examples trained on and
/// the one on examples held back. When the first fell and the second
/// rose, the model may be memorizing; that is said as a possibility.
pub fn held_out(train: (f64, f64), held: (f64, f64)) -> String {
    let t = moved("The loss on the examples it trained on", train.0, train.1);
    let h = moved("the loss on examples held back", held.0, held.1);
    let reading = match (train.1 < train.0, held.1 < held.0, held.1 > held.0) {
        (true, true, _) => "Both fell: what it learned carried over to examples it had not seen.",
        (true, _, true) => "It fit its own examples more closely and the held-back ones less: it may be memorizing them rather than learning what they share (overfitting).",
        (false, _, true) => "Neither fell: this change did not help on this task.",
        _ => "The held-back loss did not rise, so nothing here suggests memorizing.",
    };
    format!("{t}; {h}. {reading}")
}
