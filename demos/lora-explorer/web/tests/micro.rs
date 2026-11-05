//! The model, natively: the program at every rank, its stages and
//! matrices against the arithmetic done directly, the merge, the
//! update's rank, the costs at the research's sizes, and the program
//! the page shows.

use lora_explorer_web::micro::{pipeline, program, run, with, Explored, Matrix, Settings, END, PROGRAM};
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
    assert!(src.starts_with(PROGRAM[..PROGRAM.find(END).unwrap()].trim_end()), "the program itself up to the sweep, unchanged at the defaults");
    assert!(src.contains(pipeline()) && pipeline().starts_with(" c#\""), "{}", pipeline());
    assert!(rebound(&src).is_empty(), "{:?}", rebound(&src));
    assert!(program(Settings { r: 5, ..Settings::default() }).contains("\nr := 5\n"));
    assert!(at(3).tex.starts_with("\\begin{aligned}"));
}

#[test]
fn the_backward_stages_are_the_gradients() {
    for r in [1, 3, 8] {
        let e = at(r);
        let s = 6.0 / r as f64;
        assert!(e.back_same, "rank {r}: the pipeline's dx is lo:g_rads's");
        assert_eq!(e.back_stages.iter().map(|v| v.label.as_str()).collect::<Vec<_>>(), ["g", "b", "dA", "dB", ""], "rank {r}");
        let x = &e.stages[0].items;
        let h = &e.stages[1].items;
        // g = y - (W + T) x.
        let wt = Matrix { rows: 8, cols: 8, items: e.w.items.iter().zip(&e.t.items).map(|(w, t)| w + t).collect() };
        let g: Vec<f64> = e.y.iter().zip(times(&wt, x)).map(|(y, w)| y - w).collect();
        near(&e.back_stages[0].items, &g, 1e-12, "g");
        // b = B' g; dA = s b x'; dB = s g h'; dx = W' g + s A' b.
        let b: Vec<f64> = (0..r).map(|k| (0..8).map(|i| e.b.items[i * r + k] * g[i]).sum()).collect();
        near(&e.back_stages[1].items, &b, 1e-12, "b = B' g");
        assert_eq!(e.back_stages[2].shape, [r, 8]);
        let da: Vec<f64> = (0..r * 8).map(|t| s * b[t / 8] * x[t % 8]).collect();
        near(&e.back_stages[2].items, &da, 1e-12, "dA = s b x'");
        let db: Vec<f64> = (0..8 * r).map(|t| s * g[t / r] * h[t % r]).collect();
        near(&e.back_stages[3].items, &db, 1e-12, "dB = s g h'");
        let dx: Vec<f64> = (0..8).map(|j| (0..8).map(|i| e.w.items[i * 8 + j] * g[i]).sum::<f64>() + s * (0..r).map(|k| e.a.items[k * 8 + j] * b[k]).sum::<f64>()).collect();
        near(&e.back_stages[4].items, &dx, 1e-12, "dx = W' g + s A' b");
    }
}

#[test]
fn lora_reaches_the_best_possible_and_lora_fa_does_not() {
    for r in 1..=8 {
        let e = at(r);
        assert_eq!(e.loss[0], e.loss_fa[0], "rank {r}: the same start");
        assert!(e.loss.last() < e.loss.first() && e.loss_fa.last() < e.loss_fa.first(), "rank {r}: both learn");
        // T is of rank 4: the best is 0 from rank 4 on.
        assert!((e.best[3..]).iter().all(|&v| v == 0.0) && e.best[2] > 0.01);
        let (l, fa) = (*e.loss.last().unwrap(), *e.loss_fa.last().unwrap());
        assert!(l >= e.best[r - 1] - 1e-9, "rank {r}: no update of rank r does better than the SVD");
        assert!(l - e.best[r - 1] < 2e-3, "rank {r}: LoRA reaches it ({l} against {})", e.best[r - 1]);
        if r < 8 {
            assert!(fa > l + 0.02, "rank {r}: LoRA-FA ({fa}) stops above LoRA ({l})");
        }
        // The learned update is s B A of the loss: half its squared distance from T.
        let d: f64 = e.learned.items.iter().zip(&e.t.items).map(|(a, t)| (a - t) * (a - t)).sum::<f64>() / 2.0;
        assert!((d - l).abs() < 1e-9, "rank {r}: {d} against {l}");
    }
}

#[test]
fn the_sweep_follows_the_best_and_its_program_is_shown() {
    let w = lora_explorer_web::micro::sweep().unwrap();
    let e = at(3);
    for k in 0..8 {
        assert!(w.lora[k] - e.best[k] < 2e-3 && w.lora[k] >= e.best[k] - 1e-9, "rank {}: {} against {}", k + 1, w.lora[k], e.best[k]);
        assert!(w.fa[k] >= w.lora[k] - 1e-9, "rank {}: LoRA-FA no better than LoRA", k + 1);
    }
    assert!(w.fa[2] > 1.0 && w.lora[2] < 0.02, "rank 3: FA {} against LoRA {}", w.fa[2], w.lora[2]);
    let src = lora_explorer_web::micro::sweep_program();
    assert!(rebound(src).is_empty(), "{:?}", rebound(src));
    assert!(program(Settings::default()).contains(lora_explorer_web::micro::backward()));
    assert!(!program(Settings::default()).contains("### The rank sweep"));
}
