//! A callout (the Lisp timeline's, with the kinds and topics): its box,
//! the leader from the box to its month on the axis -- straight down,
//! or bent when the box has moved sideways -- and the dot where the
//! leader meets the axis.

use ml_timeline::{Event, Placed, Side};
use yew::prelude::*;

/// The event, where its box goes, the axis's height down the strip,
/// the band either side of it, and the event's place across.
#[derive(Properties, PartialEq)]
pub struct Props {
    pub event: Event,
    pub placed: Placed,
    pub axis: f64,
    pub band: f64,
    pub x: f64,
    /// Which of the four leader colors, so neighboring leaders differ.
    pub tone: usize,
}

#[function_component]
pub fn Callout(props: &Props) -> Html {
    let (axis, x, p) = (props.axis, props.x, props.placed);
    let (top, edge) = top_and_edge(&p, axis, props.band);
    let class = format!("callout {:?}", props.event.kind).to_lowercase();
    html! {
        <>
            { leader((p.left + p.width / 2.0, edge), (x, axis), (props.band, props.tone)) }
            <div class={format!("dot {}", format!("{:?}", props.event.kind).to_lowercase())} style={format!("left:{x}px;top:{axis}px")} />
            <div {class} style={format!("left:{}px;top:{top}px;width:{}px", p.left, p.width)}>
                { body(&props.event) }
            </div>
        </>
    }
}

/// Where the box's top is, and its edge nearest the axis.
fn top_and_edge(placed: &Placed, axis: f64, band: f64) -> (f64, f64) {
    let near = band + placed.offset;
    match placed.side {
        Side::Above => (axis - near - placed.height, axis - near),
        Side::Below => (axis + near, axis + near),
    }
}

/// The leader from `from`, the middle of the box's edge nearest the
/// axis, to `to`, the event on the axis: straight from the box to half
/// the band short of the axis, then on the slant. One of four tones.
fn leader(from: (f64, f64), to: (f64, f64), (band, tone): (f64, usize)) -> Html {
    let ((bx, by), (yx, ay)) = (from, to);
    let elbow = if by < ay { ay - band / 2.0 } else { ay + band / 2.0 };
    let (x0, y0) = (bx.min(yx) - 2.0, by.min(ay));
    let (w, h) = ((bx - yx).abs() + 4.0, (by - ay).abs());
    let points = format!("{},{} {},{} {},{}", bx - x0, by - y0, bx - x0, elbow - y0, yx - x0, ay - y0);
    html! {
        <svg class={format!("leader tone-{}", tone % 4)} style={format!("left:{x0}px;top:{y0}px")} width={w.to_string()} height={h.to_string()}>
            <polyline {points} />
        </svg>
    }
}

/// What the box holds: the picture, with its tip; the kind; when and
/// what; the words; the topics. All of it a link when the event leads
/// somewhere.
fn body(event: &Event) -> Html {
    let kind = match event.kind {
        ml_timeline::Kind::Paper => "Paper",
        ml_timeline::Kind::Milestone => "Milestone",
        ml_timeline::Kind::Note => "",
        ml_timeline::Kind::Demo => "Demo, live",
    };
    let inner = html! {
        <>
            if let Some(src) = &event.image {
                <div class="shot" data-tip={event.tip.clone()}><img src={src.clone()} alt={event.title.clone()} loading="lazy" /></div>
            }
            if !kind.is_empty() { <p class="kind">{ kind }</p> }
            <p class="when"><span class="date">{ &event.when }</span>{ ": " }<span class="title">{ &event.title }</span></p>
            if !event.text.is_empty() { <p class="text">{ &event.text }</p> }
            if !event.topics.is_empty() {
                <p class="topics">{ for event.topics.iter().map(|t| html! { <span class="topic">{ t }</span> }) }</p>
            }
        </>
    };
    match &event.href {
        Some(href) => html! { <a href={href.clone()} target="_blank" rel="noopener">{ inner }</a> },
        None => inner,
    }
}
