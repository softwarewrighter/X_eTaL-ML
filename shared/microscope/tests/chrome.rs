//! The pages' frame and what another repository's pages need from this
//! crate (X_eTaL-fine-tuning's asks FM1, FM2, FM5): the footer's
//! provenance by site, the libraries embedded, the heat map's scale and
//! colors.

use microscope::chrome::{provenance, Site, ML_COMMIT, THIS};
use microscope::heat::{column, fill, largest, Matrix};
use microscope::libs::names;

#[test]
fn this_repositorys_footer_names_its_commit_and_x_etal_s() {
    let p = provenance(&THIS);
    assert_eq!(p.len(), 2, "{p:?}");
    assert!(p[0].starts_with("X_eTaL ") && p[1].contains(&format!("sha {ML_COMMIT},")), "{p:?}");
    assert_eq!(THIS.url, "https://github.com/softwarewrighter/X_eTaL-ML");
}

#[test]
fn another_repositorys_footer_names_its_own_commit_and_the_x_etal_ml_it_pins() {
    const LINKS: &[(&str, &str)] = &[("All lessons", "../")];
    let site = Site { name: "X_eTaL-fine-tuning", url: "https://github.com/softwarewrighter/X_eTaL-fine-tuning", commit: "1234abc", catalog: ("../", "All lessons"), links: LINKS };
    let p = provenance(&site);
    assert_eq!(p.len(), 3, "{p:?}");
    assert_eq!(p[0], format!("X_eTaL-ML {ML_COMMIT}"));
    assert!(p[1].starts_with("X_eTaL "));
    assert!(p[2].contains("sha 1234abc,"), "{p:?}");
}

#[test]
fn the_libraries_embedded_include_this_repositorys_and_x_etal_libraries() {
    let n = names();
    for want in ["LoRA.xtl", "Optim.xtl", "Flow.xtlm", "Equations.xtlm", "Check.xtl"] {
        assert!(n.contains(&want), "{want} in {n:?}");
    }
    let mut sorted = n.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted, n, "each name once, by name");
    // Built with MICROSCOPE_LIBS naming a root, its libraries are there
    // too, and a program imports them (checked by hand with the
    // scratch library Toy; the default build names none).
    if let Some(roots) = option_env!("MICROSCOPE_LIBS") {
        if roots.contains("extra-libs") {
            assert!(n.contains(&"Toy.xtl"), "{n:?}");
            let out = microscope::run::printed("\"ty:\" u_se< \"Toy\"\nty:t_wice 21\n").unwrap();
            assert_eq!(out, ["42"]);
        }
    }
}

#[test]
fn a_library_added_at_run_time_is_imported() {
    microscope::libs::add("Added.xtl", "l:t_hrice := { x -> 3 * x }\n");
    let out = microscope::run::printed("\"ad:\" u_se< \"Added\"\nad:t_hrice 14\n").unwrap();
    assert_eq!(out, ["42"]);
}

#[test]
fn heat_maps_compare_on_one_scale_and_color_about_zero() {
    let a = Matrix { rows: 1, cols: 3, items: vec![-2.0, 0.0, 1.0] };
    let b = column(&[0.5, -4.0]);
    assert_eq!((b.rows, b.cols), (2, 1));
    assert_eq!(largest(&[&a, &b]), 4.0);
    // Red at the largest positive, blue at the largest negative, near white at 0.
    assert_eq!(fill(4.0, 4.0), "rgb(220, 38, 38)");
    assert_eq!(fill(-4.0, 4.0), "rgb(37, 99, 235)");
    assert_eq!(fill(0.0, 4.0), "rgb(245, 245, 245)");
    assert_eq!(fill(9.0, 4.0), fill(4.0, 4.0), "clamped");
    assert_eq!(fill(1.0, 0.0), fill(0.0, 4.0), "no scale: all near white");
}
