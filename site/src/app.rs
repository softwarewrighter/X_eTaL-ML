//! The page: the libraries down the side; for the one chosen, its demos
//! (editable and runnable), its reference, its source and its types.
//! The address names what is shown: #Learn/blobs.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use web_sys::HtmlTextAreaElement;
use yew::prelude::*;

use xetal_ml_libraries::render::decorated;
use xetal_ml_libraries::{expansion, expansion_marked, library, Library, GROUPS, LIBRARIES};

/// X_eTaL source in its rendered form, as a block.
fn rendered(src: &str) -> Html {
    Html::from_html_unchecked(AttrValue::from(format!("<pre class=\"xtl\">{}</pre>", decorated(src))))
}

/// An export's type line (`l:g_cd : Int -> Int -> Int`): its name
/// rendered under the library's alias, its type as written.
fn type_line(alias: &str, line: &str) -> Html {
    let (name, ty) = line.split_once(" : ").unwrap_or((line, ""));
    let name = format!("{alias}{}", name.trim_start_matches("l:"));
    let html = format!("<span class=\"tname\">{}</span> <span class=\"ttype\">: {}</span>", decorated(&name), ty.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;"));
    Html::from_html_unchecked(AttrValue::from(format!("<div class=\"typeline\">{html}</div>")))
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Demos,
    Reference,
    Source,
    Types,
}

/// What the address names: a library and one of its demos.
fn from_hash() -> (&'static Library, usize) {
    let hash = web_sys::window().and_then(|w| w.location().hash().ok()).unwrap_or_default();
    let hash = hash.trim_start_matches('#');
    let (name, demo) = hash.split_once('/').unwrap_or((hash, ""));
    let lib = library(name).unwrap_or(&LIBRARIES[0]);
    let i = lib.demos.iter().position(|d| d.name == demo).unwrap_or(0);
    (lib, i)
}

/// The address, without the #.
fn hash_now() -> String {
    let hash = web_sys::window().and_then(|w| w.location().hash().ok()).unwrap_or_default();
    hash.trim_start_matches('#').to_string()
}

/// The address of a library and demo, without the #.
fn hash_of(lib: &Library, demo: usize) -> String {
    let tail = lib.demos.get(demo).map(|d| format!("/{}", d.name)).unwrap_or_default();
    format!("{}{}", lib.name, tail)
}

fn set_hash(hash: &str) {
    if let Some(w) = web_sys::window() {
        let _ = w.location().set_hash(hash);
    }
}

#[derive(Clone, Default, PartialEq)]
struct Output {
    out: String,
    err: String,
    pictures: Vec<String>,
    ran: bool,
}

#[function_component(App)]
pub fn app() -> Html {
    let start = from_hash();
    let lib = use_state(|| start.0);
    let demo = use_state(|| start.1);
    let tab = use_state(|| Tab::Demos);
    let text = use_state(|| start.0.demos.get(start.1).map(|d| d.source.to_string()).unwrap_or_default());
    let output = use_state(Output::default);
    let seed = use_state(|| 1u64);
    let editing = use_state(|| false);
    let expanded = use_state(|| false);
    // The landing page: shown when the address names no library.
    let home = use_state(|| library(hash_now().split('/').next().unwrap_or("")).is_none());

    // The address last shown, so a change made here is not shown twice.
    let shown: Rc<RefCell<String>> = use_mut_ref(|| hash_of(start.0, start.1));
    let choose = {
        let (lib, demo, text, output, shown, home) = (lib.clone(), demo.clone(), text.clone(), output.clone(), shown.clone(), home.clone());
        move |l: &'static Library, i: usize| {
            home.set(false);
            let hash = hash_of(l, i);
            *shown.borrow_mut() = hash.clone();
            set_hash(&hash);
            lib.set(l);
            demo.set(i);
            text.set(l.demos.get(i).map(|d| d.source.to_string()).unwrap_or_default());
            output.set(Output::default());
        }
    };

    // Back, forward and links to #Library/demo show what they name.
    {
        let (choose, shown, tab, home) = (choose.clone(), shown.clone(), tab.clone(), home.clone());
        use_effect_with((), move |_| {
            let on_change = Closure::<dyn Fn()>::new(move || {
                if library(hash_now().split('/').next().unwrap_or("")).is_none() {
                    *shown.borrow_mut() = String::new();
                    home.set(true);
                    return;
                }
                let (l, i) = from_hash();
                if hash_of(l, i) != *shown.borrow() {
                    choose(l, i);
                    tab.set(Tab::Demos);
                }
            });
            let window = web_sys::window();
            if let Some(w) = &window {
                let _ = w.add_event_listener_with_callback("hashchange", on_change.as_ref().unchecked_ref());
            }
            move || {
                if let Some(w) = window {
                    let _ = w.remove_event_listener_with_callback("hashchange", on_change.as_ref().unchecked_ref());
                }
            }
        });
    }

    let run = {
        let (text, output, seed) = (text.clone(), output.clone(), seed.clone());
        Callback::from(move |_| {
            let r = xetal_play::run(&text, *seed);
            output.set(Output { out: r.out, err: r.err, pictures: r.pictures, ran: true });
        })
    };
    let reroll = {
        let seed = seed.clone();
        Callback::from(move |_| seed.set(*seed % 1_000_003 * 7919 + 17))
    };
    let toggle_edit = {
        let editing = editing.clone();
        Callback::from(move |_| editing.set(!*editing))
    };
    let toggle_expand = {
        let expanded = expanded.clone();
        Callback::from(move |_| expanded.set(!*expanded))
    };
    let reset = {
        let (lib, demo, text, output) = (lib.clone(), demo.clone(), text.clone(), output.clone());
        Callback::from(move |_| {
            text.set(lib.demos.get(*demo).map(|d| d.source.to_string()).unwrap_or_default());
            output.set(Output::default());
        })
    };
    let edit = {
        let text = text.clone();
        Callback::from(move |e: InputEvent| {
            let area: HtmlTextAreaElement = e.target_unchecked_into();
            text.set(area.value());
        })
    };

    let nav = LIBRARIES.iter().map(|l| {
        let choose = choose.clone();
        let tab = tab.clone();
        let current = !*home && l.name == lib.name;
        let onclick = Callback::from(move |_| {
            choose(l, 0);
            tab.set(Tab::Demos);
        });
        html! {
            <li class={classes!(current.then_some("current"))} {onclick}>
                <span class="name">{ l.name }</span>
                <code class="alias">{ l.alias }</code>
                <span class="summary">{ l.summary }</span>
            </li>
        }
    });

    let tabs = [(Tab::Demos, "Demos"), (Tab::Reference, "Reference"), (Tab::Source, "Source"), (Tab::Types, "Types")]
        .into_iter()
        .map(|(t, label)| {
            let tab = tab.clone();
            let current = *tab == t;
            html! { <button class={classes!("tab", current.then_some("current"))} onclick={Callback::from(move |_| tab.set(t))}>{ label }</button> }
        });

    let start_here = {
        let (home, shown) = (home.clone(), shown.clone());
        Callback::from(move |_| {
            *shown.borrow_mut() = String::new();
            set_hash("");
            home.set(true);
        })
    };

    // The landing page: what this is, how to use a library, how it fits,
    // and the libraries in groups.
    let landing = {
        let groups = GROUPS.iter().map(|(group, names)| {
            let cards = names.iter().filter_map(|n| library(n)).map(|l| {
                let choose = choose.clone();
                let tab = tab.clone();
                let onclick = Callback::from(move |_| {
                    choose(l, 0);
                    tab.set(Tab::Demos);
                });
                html! {
                    <button class="card" {onclick}>
                        <span class="name">{ l.name }<code class="alias">{ l.alias }</code></span>
                        <span class="summary">{ l.summary }</span>
                    </button>
                }
            });
            html! { <><h3>{ *group }</h3><div class="cards">{ for cards }</div></> }
        });
        html! {
            <div class="landing">
                <h2>{ "Start here" }</h2>
                <p class="lead">{ format!("{} machine-learning libraries written in X_eTaL itself, the ones the ML demos are made of: networks and their training, attention and sampling for a tiny GPT, convolution and quantized weights, classic methods and embeddings. Pick one below: its demos run here, in your browser. Press Run, then Edit the program; every library can be imported by it.", LIBRARIES.len()) }</p>
                <p>{ "In a program of your own, one line imports a library under an alias of your choice, and its functions then read like the built-ins:" }</p>
                { rendered("\"nn:\" u_se< \"NN\"\nnn:s_oftmax 1.0 2.0 3.0           # 0.09 0.24 0.67") }
                <p>{ "X_eTaL extends in three ways:" }</p>
                <table class="layers">
                    <tr><th>{ "Extends" }</th><th>{ "With" }</th><th>{ "Where" }</th></tr>
                    <tr><td>{ "the vocabulary" }</td><td>{ ".xtl libraries: functions written in X_eTaL" }</td><td><b>{ "here" }</b></td></tr>
                    <tr><td>{ "the language" }</td><td>{ ".xtlm macro libraries: source in, source out, before the program runs" }</td><td>{ "here too: Net, a network written as one line" }</td></tr>
                    <tr><td>{ "the machine" }</td><td>{ "native code behind typed X_eTaL facades" }</td><td><a href="https://github.com/softwarewrighter/X_eTaL-extensions">{ "X_eTaL-extensions" }</a></td></tr>
                </table>
                { for groups }
                <p class="more">{ "Every library's full source, cross-linked and searchable, with its demos: " }
                    <a href="../doc/index.html" target="_blank">{ "the cross-reference (xetal doc)" }</a>{ "." }
                </p>
                <p class="more">{ "The ML demos these libraries are made of: " }<a href="../">{ "the catalog" }</a>{ "." }</p>
                <p class="more">{ "More of X_eTaL: " }
                    <a href="https://softwarewrighter.github.io/X_eTaL/">{ "the language's live demo" }</a>{ ", " }
                    <a href="https://softwarewrighter.github.io/X_eTaL-demos/">{ "visual demos" }</a>{ ", " }
                    <a href="https://softwarewrighter.github.io/X_eTaL-libraries/">{ "general libraries" }</a>{ ", " }
                    <a href="https://softwarewrighter.github.io/X_eTaL-games/">{ "games" }</a>{ "." }
                </p>
            </div>
        }
    };

    let l: &'static Library = *lib;
    let body = match *tab {
        Tab::Demos => {
            let chips = l.demos.iter().enumerate().map(|(i, d)| {
                let choose = choose.clone();
                let current = i == *demo;
                html! { <button class={classes!("chip", current.then_some("current"))} onclick={Callback::from(move |_| choose(l, i))}>{ format!("{}.xtl", d.name) }</button> }
            });
            let pictures = output.pictures.iter().map(|svg| Html::from_html_unchecked(AttrValue::from(svg.clone())));
            html! {
                <div class="demos">
                    <div class="chips">{ for chips }</div>
                    if *editing {
                        <div class="edit">
                            <div class="pane">
                                <div class="label">{ "ASCII (as typed)" }</div>
                                <textarea class="editor" spellcheck="false" value={(*text).clone()} oninput={edit}
                                    rows={(text.lines().count() + 1).max(8).to_string()} />
                            </div>
                            <div class="pane">
                                <div class="label">{ "Rendered" }</div>
                                { rendered(&text) }
                            </div>
                        </div>
                    } else {
                        { rendered(&text) }
                    }
                    <div class="actions">
                        <button class="run" onclick={run}>{ "Run" }</button>
                        <button onclick={toggle_edit}>{ if *editing { "Done editing" } else { "Edit" } }</button>
                        <button onclick={reset}>{ "Reset" }</button>
                        <button onclick={toggle_expand} disabled={expansion(&text).is_none()}
                            title="the program after its macros are expanded (xetal expand)">
                            { if *expanded { "Hide expansion" } else { "Expand" } }</button>
                        <button onclick={reroll} title="a new seed for r_oll!">{ format!("Seed {}", *seed) }</button>
                        <span class="hint">{ "Edit types the program in ASCII beside its rendered form; every library here can be imported with " }{ Html::from_html_unchecked(AttrValue::from(format!("<code class=\"xtl\">{}</code>", decorated("u_se<")))) }{ "." }</span>
                    </div>
                    if *expanded {
                        { match expansion_marked(&text) {
                            Some(Ok(lines)) => {
                                let body: String = lines.iter().map(|(l, changed)| {
                                    let l = decorated(l);
                                    if *changed { format!("<span class=\"changed\">{l}</span>\n") } else { format!("{l}\n") }
                                }).collect();
                                html! { <div class="expansion">
                                    <div class="label">{ "Expanded, as xetal expand prints it: the highlighted lines are what the macro calls became" }</div>
                                    { Html::from_html_unchecked(AttrValue::from(format!("<pre class=\"xtl\">{body}</pre>"))) }
                                </div> }
                            }
                            Some(Err(e)) => html! { <pre class="err">{ e }</pre> },
                            None => html! {},
                        } }
                    }
                    if output.ran {
                        <pre class="out">{ &output.out }</pre>
                        if !output.err.is_empty() { <pre class="err">{ &output.err }</pre> }
                        <div class="pictures">{ for pictures }</div>
                    }
                </div>
            }
        }
        Tab::Reference => html! { <div class="docs">{ Html::from_html_unchecked(AttrValue::from(l.docs)) }</div> },
        Tab::Source => rendered(l.source),
        Tab::Types => html! { <div class="types">{ for l.types.lines().map(|t| type_line(l.alias, t)) }</div> },
    };

    html! {
        <>
        <header>
            <a href="../" title="All demos"><img src="xetal-logo-orange.png" alt="X_eTaL logo" /></a>
            <div>
                <h1 class="ml">{ "X_eTaL ML libraries" }</h1>
                <p>{ "The machine-learning libraries of X_eTaL-ML, written in X_eTaL: run their demos, edit them, read their references. " }<a href="../">{ "The demos" }</a>{ " are made of them." }</p>
            </div>
        </header>
        <main>
            <nav><ul>
                <li class={classes!("start", home.then_some("current"))} onclick={start_here}><span class="name">{ "Start here" }</span></li>
                { for nav }
            </ul></nav>
            if *home {
                <section>{ landing }</section>
            } else {
            <section>
                <h2>{ l.name }<code class="alias">{ l.alias }</code></h2>
                <p class="summary">{ l.summary }</p>
                <div class="import">{ rendered(&format!("{:?} u_se< {:?}", l.alias, l.name)) }</div>
                <div class="tabs">{ for tabs }</div>
                { body }
            </section>
            }
        </main>
        <footer>
            <a href="https://github.com/softwarewrighter/X_eTaL-ML">{ "Source on GitHub" }</a>
            { " | " }
            <a href="../">{ "All demos" }</a>
            { " | " }
            <a href="../doc/index.html" target="_blank">{ "Cross-reference (xetal doc)" }</a>
            { " | " }
            <a href="https://github.com/softwarewrighter/X_eTaL">{ "X_eTaL" }</a>
            { " | MIT License, Copyright (c) 2026 Michael A Wright" }
        </footer>
        </>
    }
}
