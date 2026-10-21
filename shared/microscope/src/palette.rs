//! Composing an equation or a pipeline: a palette of parts inserted
//! at the caret of an editable program, which runs in the browser;
//! what it printed (lines between `$$` drawn as math), its errors
//! placed in the text, and what its macros became. The parts are data
//! (`palette.tsv`); the model here is tested natively, the component
//! is the page's.

use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

use crate::run::expanded;
use crate::source::{block, NONE};

/// One part of the palette: its group, the label on its button, the
/// text it inserts and how far back from the text's end the caret lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub group: &'static str,
    pub label: &'static str,
    pub text: &'static str,
    pub back: usize,
}

const TSV: &str = include_str!("../palette.tsv");

/// The parts, in the file's order (a `\n` in the text is a line break).
pub fn parts() -> Vec<Part> {
    TSV.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut f = l.split('\t');
            let group = f.next().unwrap_or("");
            let label = f.next().unwrap_or("");
            let text = f.next().unwrap_or("");
            let back = f.next().and_then(|b| b.trim().parse().ok()).unwrap_or(0);
            // The text is static: a line break is written \n in the file, so
            // the text with it is leaked once (the parts are read once per page).
            let text: &'static str = if text.contains("\\n") { Box::leak(text.replace("\\n", "\n").into_boxed_str()) } else { text };
            Part { group, label, text, back }
        })
        .collect()
}

/// The groups, in the order they first appear.
pub fn groups(parts: &[Part]) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for p in parts {
        if !out.contains(&p.group) {
            out.push(p.group);
        }
    }
    out
}

/// `part` put into `text` over the selection `start..end` (character
/// positions): the new text and where the caret lands.
pub fn insert(text: &str, start: usize, end: usize, part: &Part) -> (String, usize) {
    let chars: Vec<char> = text.chars().collect();
    let (start, end) = (start.min(chars.len()), end.min(chars.len()));
    let (start, end) = (start.min(end), start.max(end));
    let mut out: String = chars[..start].iter().collect();
    out.push_str(part.text);
    out.extend(chars[end..].iter());
    let caret = start + part.text.chars().count().saturating_sub(part.back);
    (out, caret)
}

/// What composing gave: the lines printed (those between `$$` taken
/// out as LaTeX), the errors and warnings as X_eTaL places them in the
/// text (`--> page:LINE:COL`, the line marked), and the program after
/// its macros were expanded.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Composed {
    pub out: Vec<String>,
    pub tex: Vec<String>,
    pub err: String,
    pub expansion: Option<Result<String, String>>,
}

/// Run `src` (seed 1) and expand it.
pub fn compose(src: &str) -> Composed {
    crate::libs::install();
    let run = xetal_play::run(src, 1);
    let (tex, out): (Vec<String>, Vec<String>) =
        run.out.lines().map(str::to_string).partition(|l| l.len() > 4 && l.starts_with("$$") && l.ends_with("$$"));
    let tex = tex.iter().map(|l| l[2..l.len() - 2].to_string()).collect();
    let expansion = (src.contains('<') && src.contains("u_se<")).then(|| expanded(src));
    Composed { out, tex, err: run.err.trim_end().to_string(), expansion }
}

#[derive(Properties, PartialEq)]
pub struct Props {
    /// The programs to start from: a title and the source of each.
    pub starters: Vec<(&'static str, &'static str)>,
}

/// The composer: the starters, the palette, the editable program (ASCII
/// as typed beside its rendered form), Run and Expand, and what the
/// run gave.
#[function_component(Composer)]
pub fn composer(props: &Props) -> Html {
    let parts = use_memo((), |_| parts());
    let text = use_state(|| props.starters.first().map_or(String::new(), |s| s.1.to_string()));
    let result = use_state(|| None::<Composed>);
    let showing = use_state(|| false);
    let area = use_node_ref();

    let edit = {
        let text = text.clone();
        Callback::from(move |e: InputEvent| {
            let a: HtmlTextAreaElement = e.target_unchecked_into();
            text.set(a.value());
        })
    };
    let run = {
        let (text, result) = (text.clone(), result.clone());
        Callback::from(move |_| result.set(Some(compose(&text))))
    };
    let toggle = {
        let showing = showing.clone();
        Callback::from(move |_| showing.set(!*showing))
    };
    let starters = props.starters.iter().map(|(title, src)| {
        let (text, result) = (text.clone(), result.clone());
        let src = *src;
        html! { <button onclick={Callback::from(move |_| { text.set(src.to_string()); result.set(None); })}>{title}</button> }
    });
    let palette = groups(&parts).into_iter().map(|g| {
        let buttons = parts.iter().filter(|p| p.group == g).map(|p| {
            let (text, area) = (text.clone(), area.clone());
            let p = *p;
            let onclick = Callback::from(move |_| {
                let Some(a) = area.cast::<HtmlTextAreaElement>() else { return };
                let start = a.selection_start().ok().flatten().unwrap_or(0) as usize;
                let end = a.selection_end().ok().flatten().unwrap_or(start as u32) as usize;
                let (new, caret) = insert(&a.value(), start, end, &p);
                a.set_value(&new);
                let _ = a.set_selection_range(caret as u32, caret as u32);
                let _ = a.focus();
                text.set(new);
            });
            html! { <button class="part" title={p.text} {onclick}>{p.label}</button> }
        });
        html! { <div class="group"><span class="name">{g}</span>{ for buttons }</div> }
    });
    let rows = (text.lines().count() + 1).max(8).to_string();
    let gave = result.as_ref().map(|r| html! { <>
        { for r.tex.iter().map(|t| html! { <div class="tex" data-tex={t.clone()}>{t.clone()}</div> }) }
        if !r.out.is_empty() { <pre class="output">{ r.out.join("\n") }</pre> }
        if !r.err.is_empty() { <pre class="err">{ r.err.clone() }</pre> }
        if *showing {
            { match &r.expansion {
                Some(Ok(x)) => html! { <div class="expansion"><div class="label">{"Expanded, as xetal expand prints it"}</div>{ block(x, NONE) }</div> },
                Some(Err(e)) => html! { <pre class="err">{ e.clone() }</pre> },
                None => html! { <p class="note">{"Nothing to expand: the program calls no macro."}</p> },
            } }
        }
    </> });
    html! {
        <section class="panel compose">
            <h2>{"Compose"}</h2>
            <p class="note">{"Start from a program, then type, or put a part at the caret from the palette: the equation's grammar (as a paper prints it), the pipeline's, a library's functions as stages, the imports. Run shows what it printed (a line between $$ drawn as math), any error placed in the text, and Expand what the macros became. The macro's rule, every part of its text used or the call refused, is the lint of the notation."}</p>
            <div class="controls"><span>{"Start from: "}</span>{ for starters }</div>
            <div class="palette">{ for palette }</div>
            <div class="edit">
                <div class="pane">
                    <div class="label">{"ASCII (as typed)"}</div>
                    <textarea class="editor" ref={area} spellcheck="false" value={(*text).clone()} oninput={edit} {rows} />
                </div>
                <div class="pane">
                    <div class="label">{"Rendered"}</div>
                    { block(&text, NONE) }
                </div>
            </div>
            <div class="controls">
                <button class="run" onclick={run}>{"Run"}</button>
                <button onclick={toggle}>{ if *showing { "Hide expansion" } else { "Expand" } }</button>
            </div>
            { gave.unwrap_or_default() }
        </section>
    }
}
