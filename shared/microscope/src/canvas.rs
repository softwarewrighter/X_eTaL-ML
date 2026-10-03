//! A canvas showing an RGBA image, scaled to its box, reporting clicks
//! as (row, column).

use std::rc::Rc;

use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};
use yew::prelude::*;

#[derive(Properties, PartialEq)]
pub struct Props {
    pub rows: usize,
    pub cols: usize,
    pub rgba: Rc<Vec<u8>>,
    #[prop_or_default]
    pub mark: Option<(usize, usize)>,
    #[prop_or_default]
    pub onclick: Option<Callback<(usize, usize)>>,
    #[prop_or_default]
    pub class: Classes,
}

fn draw(canvas: &HtmlCanvasElement, p: &Props) -> Option<()> {
    let ctx: CanvasRenderingContext2d = canvas.get_context("2d").ok()??.dyn_into().ok()?;
    let mut data = (*p.rgba).clone();
    if let Some((y, x)) = p.mark {
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)] {
            let (yy, xx) = (y as i64 + dy, x as i64 + dx);
            if (0..p.rows as i64).contains(&yy) && (0..p.cols as i64).contains(&xx) {
                let i = 4 * (yy as usize * p.cols + xx as usize);
                data[i..i + 3].copy_from_slice(&[255, 64, 160]);
            }
        }
    }
    let img = ImageData::new_with_u8_clamped_array_and_sh(Clamped(&data), p.cols as u32, p.rows as u32).ok()?;
    ctx.put_image_data(&img, 0.0, 0.0).ok()
}

#[function_component(Canvas)]
pub fn canvas(p: &Props) -> Html {
    let node = use_node_ref();
    {
        let node = node.clone();
        let deps = (p.rgba.clone(), p.mark, p.rows, p.cols);
        let props = Props { rows: p.rows, cols: p.cols, rgba: p.rgba.clone(), mark: p.mark, onclick: None, class: Classes::new() };
        use_effect_with(deps, move |_| {
            if let Some(c) = node.cast::<HtmlCanvasElement>() {
                draw(&c, &props);
            }
        });
    }
    let onclick = p.onclick.clone().map(|cb| {
        let (rows, cols) = (p.rows, p.cols);
        Callback::from(move |e: MouseEvent| {
            let Some(el) = e.target_dyn_into::<HtmlCanvasElement>() else { return };
            let r = el.get_bounding_client_rect();
            let x = ((e.client_x() as f64 - r.left()) / r.width() * cols as f64) as usize;
            let y = ((e.client_y() as f64 - r.top()) / r.height() * rows as f64) as usize;
            cb.emit((y.min(rows - 1), x.min(cols - 1)));
        })
    });
    let class = classes!("pic", p.class.clone(), p.onclick.is_some().then_some("clickable"));
    html! { <canvas ref={node} {class} width={p.cols.to_string()} height={p.rows.to_string()} {onclick} /> }
}
