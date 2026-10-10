fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<embedding_explorer_web::app::App>::new().render();
}
