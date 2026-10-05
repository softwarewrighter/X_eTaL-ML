fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<net_macro_web::app::App>::new().render();
}
