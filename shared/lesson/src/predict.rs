//! A question answered before the experiment is shown (docs/plan.md,
//! principle 1: every lesson opens with a prediction). The reader picks
//! an answer, or says they would rather just see; only then is the
//! experiment drawn, with which answer the run bears out. The right
//! answer is given by the page from its run, not fixed in the text.

use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct PredictProps {
    /// The question.
    pub question: AttrValue,
    /// The answers offered.
    pub choices: Vec<AttrValue>,
    /// The one the run bears out (an index into `choices`), from the
    /// page's own run.
    pub right: usize,
    /// Why, in a sentence, said once an answer is in.
    #[prop_or_default]
    pub why: AttrValue,
    /// The experiment, drawn once an answer is in.
    #[prop_or_default]
    pub children: Html,
}

/// What the reader has done: nothing yet, picked an answer, or asked
/// to see without predicting.
#[derive(Clone, Copy, PartialEq)]
enum Answer {
    None,
    Picked(usize),
    Skipped,
}

/// What is said once an answer is in.
pub fn verdict(picked: Option<usize>, right: usize, choices: &[AttrValue]) -> String {
    let truth = choices.get(right).map(|c| c.to_string()).unwrap_or_default();
    match picked {
        Some(p) if p == right => format!("You predicted: {truth}. The run bears that out."),
        Some(p) => format!("You predicted: {}. The run gives: {truth}.", choices.get(p).map(|c| c.to_string()).unwrap_or_default()),
        None => format!("The run gives: {truth}."),
    }
}

#[function_component(Predict)]
pub fn predict(props: &PredictProps) -> Html {
    let answer = use_state(|| Answer::None);
    let now = *answer;
    let choices = props.choices.iter().enumerate().map(|(i, c)| {
        let answer = answer.clone();
        let pick = Callback::from(move |_| answer.set(Answer::Picked(i)));
        html! { <li><button type="button" class={classes!("choice", (now == Answer::Picked(i)).then_some("on"))} onclick={pick} disabled={now != Answer::None}>{ c.clone() }</button></li> }
    });
    let skip = { let answer = answer.clone(); Callback::from(move |_| answer.set(Answer::Skipped)) };
    let again = { let answer = answer.clone(); Callback::from(move |_| answer.set(Answer::None)) };
    html! {
        <section class="predict">
            <p class="question"><b>{"Predict first: "}</b>{ props.question.clone() }</p>
            <ul class="choices">{ for choices }</ul>
            {
                match *answer {
                    Answer::None => html! { <button type="button" class="skip" onclick={skip}>{"Just show me"}</button> },
                    done => {
                        let picked = if let Answer::Picked(p) = done { Some(p) } else { None };
                        html! { <>
                            <p class="verdict" role="status">{ verdict(picked, props.right, &props.choices) }{" "}{ props.why.clone() }</p>
                            <button type="button" class="skip" onclick={again}>{"Ask me again"}</button>
                            { props.children.clone() }
                        </> }
                    }
                }
            }
        </section>
    }
}
