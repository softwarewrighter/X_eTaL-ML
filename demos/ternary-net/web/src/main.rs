fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<ternary_net_web::app::App>::new().render();
}
