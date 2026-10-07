fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<train_live_web::app::App>::new().render();
}
