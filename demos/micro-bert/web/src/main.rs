fn main() {
    console_error_panic_hook::set_once();
    yew::Renderer::<micro_bert_web::app::App>::new().render();
}
