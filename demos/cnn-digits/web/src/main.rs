fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<cnn_digits_web::app::App>::new().render();
}
