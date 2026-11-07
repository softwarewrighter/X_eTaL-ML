//! A page opened at a given state: `?stage=3&rank=5&mode=advanced` in
//! its address sets where it starts, so a state can be linked to (from
//! another lesson, or a README) and so the browser check can look at
//! more than the page as first loaded. A page reads the settings it
//! knows; anything else, or a value out of range, is ignored.

/// The value of `name` in a query string (`?a=1&b=2`, the `?` optional).
pub fn value(query: &str, name: &str) -> Option<String> {
    query.trim_start_matches('?').split('&').filter_map(|pair| pair.split_once('=')).find(|(k, _)| *k == name).map(|(_, v)| v.to_string())
}

/// The value of `name` in the page's address, in the browser; None
/// elsewhere (a native test).
pub fn param(name: &str) -> Option<String> {
    #[cfg(target_arch = "wasm32")]
    {
        let query = web_sys::window()?.location().search().ok()?;
        value(&query, name)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = name;
        None
    }
}

/// A whole number from the address, kept only inside `low..=high`.
pub fn number(name: &str, low: usize, high: usize) -> Option<usize> {
    param(name)?.parse().ok().filter(|n| (low..=high).contains(n))
}
