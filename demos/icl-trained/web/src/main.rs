fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<icl_trained_web::app::App>::new().render();
}
