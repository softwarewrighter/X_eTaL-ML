fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<k_means_web::app::App>::new().render();
}
