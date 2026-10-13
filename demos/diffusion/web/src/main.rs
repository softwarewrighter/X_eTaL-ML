fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<diffusion_web::app::App>::new().render();
}
