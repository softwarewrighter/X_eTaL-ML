//! X_eTaL source drawn decorated (as X_eTaL renders it, colored), with
//! a range highlighted as one continuous block, and shapes as s_hape.

use xetal_play::{decorate, Class};
use yew::prelude::*;

/// A byte range of a source; `NONE` highlights nothing.
pub type Range = (usize, usize);
pub const NONE: Range = (1, 0);

fn css(class: Class) -> &'static str {
    match class {
        Class::Builtin => "t-builtin",
        Class::UserFunc | Class::LibFunc | Class::Macro => "t-user",
        Class::LambdaArg => "t-arg",
        Class::Number | Class::Exponent => "t-num",
        Class::Symbol | Class::Quote => "t-sym",
        Class::Comment => "t-comment",
        _ => "t-plain",
    }
}

fn segments(src: &str, (lo, hi): Range) -> Html {
    // Runs of segments inside lo..hi are wrapped in one highlight block.
    let mut out: Vec<Html> = Vec::new();
    let mut run: Vec<Html> = Vec::new();
    let flush = |run: &mut Vec<Html>, out: &mut Vec<Html>| {
        if !run.is_empty() {
            let inner = std::mem::take(run);
            out.push(html! { <span class="hl">{ for inner }</span> });
        }
    };
    for s in decorate(src) {
        let inside = s.raw.start >= lo && s.raw.end <= hi && s.class != Class::Comment;
        let seg = html! { <span class={css(s.class)}>{s.text}</span> };
        if inside {
            run.push(seg);
        } else {
            flush(&mut run, &mut out);
            out.push(seg);
        }
    }
    flush(&mut run, &mut out);
    html! { <>{ for out }</> }
}

/// Any X_eTaL snippet, drawn decorated, inline.
pub fn code(src: &str) -> Html {
    html! { <code class="xtl">{ segments(src, NONE) }</code> }
}

/// One line of X_eTaL, large, `range` highlighted.
pub fn line(src: &str, range: Range) -> Html {
    html! { <code class="line">{ segments(src, range) }</code> }
}

/// A program (several lines, with comments), `range` highlighted.
pub fn block(src: &str, range: Range) -> Html {
    html! { <pre class="source">{ segments(src, range) }</pre> }
}

/// The range of `src` from `from` to the end of the line holding the
/// first `to` after it.
pub fn between(src: &str, from: &str, to: &str) -> Range {
    let a = src.find(from).unwrap_or(0);
    let b = src[a..].find(to).map_or(a, |i| a + i + to.len());
    let end = src[b..].find('\n').map_or(src.len(), |i| b + i);
    (a, end)
}

/// The range of the first `part` in `src` (`NONE` if it is not there).
pub fn find(src: &str, part: &str) -> Range {
    src.find(part).map_or(NONE, |i| (i, i + part.len()))
}

/// A shape as X_eTaL's s_hape gives it, decorated, with a tooltip
/// saying what the axes are; a scalar's shape is empty.
pub fn shape(dims: &[usize], meaning: &str) -> Html {
    let text: Vec<String> = dims.iter().map(usize::to_string).collect();
    let shown = match text.is_empty() {
        true => html! { <>{ code("s_hape") }{ " is empty: a scalar" }</> },
        false => code(&format!("s_hape = {}", text.join(" "))),
    };
    html! { <span class="shape" title={format!("the array's shape (s_hape): {meaning}")}>{ shown }</span> }
}
