//! How much a page shows: three depths over one computation (docs/plan.md,
//! principle 2). Beginner is guided and in plain words; Explorer adds
//! every control and the short explanations; Advanced adds the
//! equations, the gradients and the code. The mode is the page's own
//! state, apart from the experiment's, so switching it never resets or
//! reruns anything.

use yew::prelude::*;

/// A depth of explanation; each shows all that the one before it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Beginner,
    Explorer,
    Advanced,
}

impl Mode {
    pub const ALL: [Mode; 3] = [Mode::Beginner, Mode::Explorer, Mode::Advanced];

    /// Its name on the switch.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Beginner => "Beginner",
            Mode::Explorer => "Explorer",
            Mode::Advanced => "Advanced",
        }
    }

    /// What it shows, in a sentence, for the reader choosing.
    pub fn about(self) -> &'static str {
        match self {
            Mode::Beginner => "A guided path in plain words, one thing at a time. Nothing is assumed.",
            Mode::Explorer => "Everything on the page at once, with every control and a short explanation of each.",
            Mode::Advanced => "Adds the equations written out, the gradients and the code that computes it all.",
        }
    }

    /// The mode named in a page's address (`?mode=explorer`), if any.
    pub fn from_link() -> Option<Mode> {
        let named = crate::link::param("mode")?;
        Mode::ALL.into_iter().find(|m| m.label().eq_ignore_ascii_case(&named))
    }

    /// Whether a part meant for readers at `depth` and deeper is shown.
    pub fn shows(self, depth: Mode) -> bool {
        self >= depth
    }
}

/// `body` when the page's mode shows parts of `depth`, nothing otherwise.
pub fn when(mode: Mode, depth: Mode, body: Html) -> Html {
    if mode.shows(depth) { body } else { html! {} }
}

#[derive(Properties, PartialEq)]
pub struct SwitchProps {
    pub mode: Mode,
    pub onchange: Callback<Mode>,
}

/// The switch: three choices, one of them on, and what the one on
/// shows. Radio buttons, so it works by keyboard (arrows) and touch.
#[function_component(Switch)]
pub fn switch(props: &SwitchProps) -> Html {
    let choices = Mode::ALL.iter().map(|m| {
        let (m, onchange) = (*m, props.onchange.clone());
        let pick = Callback::from(move |_| onchange.emit(m));
        html! {
            <label class={classes!("mode", (props.mode == m).then_some("on"))}>
                <input type="radio" name="lesson-mode" checked={props.mode == m} onchange={pick} />
                { m.label() }
            </label>
        }
    });
    html! {
        <fieldset class="modes">
            <legend>{"How much to show"}</legend>
            { for choices }
            <p class="mode-about" role="status">{ props.mode.about() }</p>
        </fieldset>
    }
}
