fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<world_model_web::app::App>::new().render();
}
