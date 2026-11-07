//! Arithmetic written out, so no number on a page appears from nowhere:
//! one cell of a product as a sum in symbols, then with the run's
//! numbers. Presentation only: the products and the total are the ones
//! X_eTaL printed; nothing here multiplies or adds for the page (the
//! check that they add up is for tests).

use yew::prelude::*;

/// A number as a page shows it: six decimals at most, no trailing zeros.
pub fn show(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".into() } else { s.to_string() }
}

/// A count with commas between its thousands.
pub fn commas(n: usize) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Numbers in a row, with commas between.
pub fn list(v: &[f64]) -> String {
    v.iter().map(|x| show(*x)).collect::<Vec<_>>().join(", ")
}

/// A number as a term of a sum: a negative one in brackets, so that no
/// two signs stand side by side (`3 + (-1)`, not `3 + -1`).
pub fn term(v: f64) -> String {
    if v < 0.0 { format!("({})", show(v)) } else { show(v) }
}

/// One product as text: `a x b`, a negative factor in brackets.
pub fn product(a: f64, b: f64) -> String {
    format!("{} \u{00d7} {}", term(a), term(b))
}

/// The sum as text: `a1 x b1 + a2 x b2 + ... = p1 + p2 + ... = total`,
/// the factors, the products and the total all as given.
pub fn written(name: &str, a: &[f64], b: &[f64], products: &[f64], total: f64) -> String {
    let terms: Vec<String> = a.iter().zip(b).map(|(a, b)| product(*a, *b)).collect();
    let sums: Vec<String> = products.iter().map(|v| term(*v)).collect();
    format!("{name} = {} = {} = {}", terms.join(" + "), sums.join(" + "), show(total))
}

/// The sum as text with every number rounded to `places` decimals, for
/// a sum of many terms: `a1 x b1 + ... = p1 + ... = total`. The page
/// says the numbers are rounded; the full ones are in the pictures.
pub fn written_to(places: usize, name: &str, a: &[f64], b: &[f64], products: &[f64], total: f64) -> String {
    let cut = |v: f64| {
        let p = 10f64.powi(places as i32);
        (v * p).round() / p
    };
    let round = |v: &[f64]| v.iter().map(|x| cut(*x)).collect::<Vec<_>>();
    written(name, &round(a), &round(b), &round(products), cut(total))
}

/// `v` rounded to `places` decimals, for showing (the full number is
/// in the picture's inspector).
pub fn rounded(v: f64, places: usize) -> f64 {
    let p = 10f64.powi(places as i32);
    (v * p).round() / p
}

/// A sum written out with its numbers, as a line of a page.
pub fn numeric(name: &str, a: &[f64], b: &[f64], products: &[f64], total: f64) -> Html {
    html! { <p class="worked">{ written(name, a, b, products, total) }</p> }
}

/// The same sum in symbols, for one row of a table times a list:
/// `y(row) = W(row,1) x(1) + W(row,2) x(2) + ...`, rows and columns
/// counted from 1 as the page's tables label them.
pub fn in_symbols(result: &str, table: &str, list: &str, row: usize, n: usize) -> String {
    let terms: Vec<String> = (1..=n).map(|j| format!("{table}({row},{j}) \u{00d7} {list}({j})")).collect();
    format!("{result}({row}) = {}", terms.join(" + "))
}

/// The sum in symbols, as a line of a page.
pub fn symbolic(result: &str, table: &str, list: &str, row: usize, n: usize) -> Html {
    html! { <p class="worked symbols">{ in_symbols(result, table, list, row, n) }</p> }
}

/// Whether the products add up to the total, within `tolerance`: for a
/// page's test, that what it writes out is what X_eTaL computed.
pub fn adds_up(products: &[f64], total: f64, tolerance: f64) -> bool {
    (products.iter().sum::<f64>() - total).abs() <= tolerance
}
