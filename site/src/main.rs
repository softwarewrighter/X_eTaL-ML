//! The browser app: run with trunk (just serve, just pages).

mod app;

fn main() {
    console_error_panic_hook::set_once();
    xetal_ml_libraries::store::install();
    // []TS reads the browser's clock (JavaScript's Date): without it the
    // host has none and Dates' t_oday is error[no-clock].
    xetal_clock::install(std::sync::Arc::new(xetal_webclock::Browser));
    yew::Renderer::<app::App>::new().render();
}
