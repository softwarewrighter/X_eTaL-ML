//! The model, natively: the program at every rank, its stages and
//! matrices against the arithmetic done directly, the merge, the
//! update's rank, the costs at the research's sizes, and the program
//! the page shows.

use lora_explorer_web::micro::{pipeline, program, run, with, Explored, Matrix, Settings, PROGRAM};
use microscope::source::rebound;

/// M v for a matrix and a vector.
fn times(m: &Matrix, v: &[f64]) -> Vec<f64> {
    (0..m.rows).map(|i| (0..m.cols).map(|j| m.items[i * m.cols + j] * v[j]).sum()).collect()
}

fn near(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{what}: {x} != {y}");
    }
}

fn at(r: usize) -> Explored {
    run(Settings { r, ..Settings::default() }).unwrap_or_else(|e| panic!("rank {r}: {e}"))
}

#[test]
fn every_rank_runs_and_the_stages_are_the_arithmetic() {
    for r in 1..=8 {
        let e = at(r);
        let x = &e.stages[0].items;
        let h = times(&e.a, x);
        let u = times(&e.b, &h);
        let s = 6.0 / r as f64;
        let y: Vec<f64> = times(&e.w, x).iter().zip(&u).map(|(w, u)| w + s * u).collect();
        assert_eq!(e.stages.iter().map(|v| v.label.as_str()).collect::<Vec<_>>(), ["x", "h", "u", ""], "rank {r}");
        assert_eq!(e.stages[1].shape, [r]);
        near(&e.stages[1].items, &h, 1e-12, "h = A x");
        near(&e.stages[2].items, &u, 1e-12, "u = B h");
        near(&e.stages[3].items, &y, 1e-12, "y = W x + s u");
        near(&e.y, &y, 1e-12, "the program's y");
    }
}

#[test]
fn the_update_has_the_rank_and_the_merge_is_the_same_layer() {
    for r in 1..=8 {
        let e = at(r);
        assert_eq!(e.rank, r, "the update's rank");
        assert!(e.merged_same, "rank {r}");
        let merged: Vec<f64> = e.w.items.iter().zip(&e.delta.items).map(|(w, d)| w + d).collect();
        near(&e.merged.items, &merged, 1e-15, "W + s B A");
        near(&times(&e.merged, &e.stages[0].items), &e.y, 1e-12, "(W + s B A) x");
        assert_eq!(e.trained, (16 * r, 64));
        assert_eq!(e.costs, (16 * r, 64 * r));
    }
}

#[test]
fn the_costs_are_the_researchs_worked_numbers_at_rank_16() {
    // The slider stops at 8 (an 8 x 8 matrix); the sizes panel's sums
    // are r's, so the research's rank 16 is checked through the program.
    let src = with(Settings::default()).replacen("r := 3\n", "r := 16\n", 1);
    let p = &src[..src.find("### The update").unwrap()];
    let tail = "n := r lo:e_xpert d c_at f\nmib := (16 lo:b_ytes n) / 1048576.0\nn\nmib c_at (mib * f_loat experts) c_at mib * f_loat experts * layers\n";
    // Only the data and the sizes run (the layer is not needed).
    let sizes = format!("{}{tail}", &p[..p.find("### The adapted layer").unwrap()]);
    let out = microscope::run::printed(&sizes).unwrap();
    assert_eq!(out[0], "884736");
    let mib: Vec<f64> = out[1].split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert!((mib[0] - 1.6875).abs() < 1e-12 && (mib[1] - 108.0).abs() < 1e-9);
    assert!(((mib[2] / 1024.0) * 100.0).round() / 100.0 == 2.53);
    // And at the default rank 3: 3 r (d + f).
    assert_eq!(at(3).expert, 3 * 3 * (4096 + 14336));
}

#[test]
fn the_sizes_are_the_pages() {
    let e = run(Settings { r: 2, d: 100, f: 300, experts: 4, layers: 5 }).unwrap();
    assert_eq!(e.expert, 3 * 2 * 400);
    let mib = 2400.0 * 2.0 / 1048576.0;
    near(&[e.mib.0, e.mib.1, e.mib.2], &[mib, 4.0 * mib, 20.0 * mib], 1e-15, "MiB");
}

#[test]
fn the_page_shows_the_command_line_programs_code() {
    let src = program(Settings::default());
    assert!(src.starts_with(PROGRAM.trim_end()), "the program itself, unchanged at the defaults");
    assert!(src.contains(pipeline()) && pipeline().starts_with(" c#\""), "{}", pipeline());
    assert!(rebound(&src).is_empty(), "{:?}", rebound(&src));
    assert!(program(Settings { r: 5, ..Settings::default() }).contains("\nr := 5\n"));
    assert!(at(3).tex.starts_with("\\begin{aligned}"));
}
