//! Small arrays drawn as grids of cells (HTML), for boards a person
//! clicks on cell by cell: 0 / 1 boards, and counts shaded with numbers.

use yew::prelude::*;

/// How a grid's cells are colored.
#[derive(Clone, Copy, PartialEq)]
pub enum Paint {
    /// 0 / 1 cells.
    Cells,
    /// 0..9 sums, shaded, numbers shown.
    Sum,
}

pub struct Grid<'a> {
    pub cells: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub paint: Paint,
    pub selected: Option<(usize, usize)>,
    pub size: &'static str,
    pub onclick: Option<Callback<(usize, usize)>>,
}

fn cell(g: &Grid, y: usize, x: usize) -> Html {
    let v = g.cells[y * g.cols + x];
    let mut class = classes!("cell");
    if g.selected == Some((y, x)) {
        class.push("sel");
    }
    let (style, text) = match g.paint {
        Paint::Cells => {
            class.push(if v == 1 { "on" } else { "off" });
            (String::new(), String::new())
        }
        Paint::Sum => {
            let text = if v == 0 { String::new() } else { v.to_string() };
            (format!("--heat:{}", v.min(9)), text)
        }
    };
    if g.paint == Paint::Sum {
        class.push("heat");
        class.push(match v { 3 => "s3", 4 => "s4", _ => "" });
        if v >= 5 {
            class.push("hot");
        }
    }
    let onclick = g.onclick.clone().map(|cb| Callback::from(move |_: MouseEvent| cb.emit((y, x))));
    html! { <div {class} {style} {onclick}>{text}</div> }
}

pub fn grid(g: Grid) -> Html {
    let style = format!("grid-template-columns: repeat({}, 1fr)", g.cols);
    let cells = (0..g.rows).flat_map(|y| (0..g.cols).map(move |x| (y, x)));
    html! {
        <div class={classes!("grid", g.size)} {style}>
            { for cells.map(|(y, x)| cell(&g, y, x)) }
        </div>
    }
}
