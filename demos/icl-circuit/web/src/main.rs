fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<icl_circuit_web::app::App>::new().render();
}
