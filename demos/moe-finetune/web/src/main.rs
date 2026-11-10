fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<moe_finetune_web::app::App>::new().render();
}
