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
    let e = run(Settings { r: 2, d: 100, f: 300, experts: 4, layers: 5, qk: 16 }).unwrap();
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
        assert!(e.back_same, "rank {r}: the pipeline's dx is lo:a_dapter<'s");
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

/// QLoRA's NF4 levels (Dettmers et al. 2023, appendix E).
const NF4: [f64; 16] = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0];

#[test]
fn w_in_4_bits_is_the_blockwise_arithmetic_at_every_block() {
    for qk in [8, 16, 32, 64] {
        let e = run(Settings { qk, ..Settings::default() }).unwrap_or_else(|err| panic!("block {qk}: {err}"));
        let w = &e.w.items;
        let mut u = vec![0.0; 64];
        let mut n = vec![0.0; 64];
        for b in w.chunks(qk).enumerate() {
            let (i, block) = b;
            let scale = block.iter().fold(0.0f64, |a, v| a.max(v.abs()));
            for (j, v) in block.iter().enumerate() {
                let step = scale / 7.0;
                u[i * qk + j] = (v / step + 0.5).floor().clamp(-7.0, 7.0) * step;
                let x = v / scale;
                let near = NF4.iter().copied().fold(NF4[0], |best, l| if (l - x).abs() < (best - x).abs() { l } else { best });
                n[i * qk + j] = near * scale;
            }
        }
        near(&e.w_uniform.items, &u, 1e-12, "uniform 4 bits");
        near(&e.w_nf4.items, &n, 1e-12, "NF4");
        let rmse = |q: &[f64]| (q.iter().zip(w).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / 64.0).sqrt();
        near(&e.rmse[..2], &[rmse(&u), rmse(&n)], 1e-12, "the RMSE on W");
        assert!(e.rmse[0] < e.rmse[1], "block {qk}: on W (a sine) uniform wins: {:?}", e.rmse);
        assert_eq!(e.bits.0, 4.0 + 32.0 / qk as f64);
        assert!((e.bits.1 - (4.0 + 8.0 / qk as f64 + 64.0 / (qk as f64 * 256.0))).abs() < 1e-12);
        near(&e.big_mib[..2], &[224.0, 112.0], 1e-9, "FP32 and FP16 MiB");
        assert!(e.moved.0 > 0.0 && e.moved.1 > 0.0, "a 4-bit base moves the output");
    }
    let e = at(3);
    assert!(e.rmse[3] < e.rmse[2], "on normal numbers NF4 wins: {:?}", e.rmse);
    // At QLoRA's block of 64: 4.5 bits, 31.5 MiB for the projection.
    let q = run(Settings { qk: 64, ..Settings::default() }).unwrap();
    assert_eq!(q.bits.0, 4.5);
    assert!((q.big_mib[2] - 31.5).abs() < 1e-9 && (q.big_mib[3] - 28.9).abs() < 0.05, "{:?}", q.big_mib);
}

#[test]
fn qlora_reaches_the_best_for_what_is_left_and_its_merge_is_not_exact() {
    for qk in [16, 64] {
        let e = run(Settings { qk, ..Settings::default() }).unwrap();
        let l = *e.loss_q.last().unwrap();
        assert!(l >= e.best_q - 1e-9 && l < e.best_q + 2e-3, "block {qk}: QLoRA {l} against its best {}", e.best_q);
        assert!(e.best_q > e.best[2], "block {qk}: the 4-bit base leaves more to learn");
        assert!(e.merge_miss.0 > 1e-6 && e.merge_miss.1 > 1e-6, "block {qk}: the merge into 4 bits is not exact: {:?}", e.merge_miss);
    }
}

#[test]
fn vera_s_stages_are_the_arithmetic_and_the_macro_s() {
    for r in [1, 3, 8] {
        let e = at(r);
        assert_eq!(e.v_same, (true, true), "rank {r}: VeRA's pipelines are lo:a_dapter<'s");
        assert_eq!(e.v_stages.iter().map(|v| v.label.as_str()).collect::<Vec<_>>(), ["x", "h", "u", "v", ""], "rank {r}");
        assert_eq!(e.v_back_stages.iter().map(|v| v.label.as_str()).collect::<Vec<_>>(), ["g", "c", "dd", "db", ""], "rank {r}");
        // The made-up vectors, as the program makes them (r_ange from 1).
        let dv: Vec<f64> = (1..=r).map(|k| 0.5 + 0.3 * (0.8 * k as f64).cos()).collect();
        let bv: Vec<f64> = (1..=8).map(|i| 0.6 * (1.7 * i as f64).sin()).collect();
        let x = &e.stages[0].items;
        let h = times(&e.a0, x);
        let u: Vec<f64> = dv.iter().zip(&h).map(|(d, h)| d * h).collect();
        let v = times(&e.b0, &u);
        let y: Vec<f64> = times(&e.w, x).iter().zip(&v).zip(&bv).map(|((w, v), b)| w + b * v).collect();
        near(&e.v_stages[1].items, &h, 1e-12, "h = A0 x");
        near(&e.v_stages[2].items, &u, 1e-12, "u = d h");
        near(&e.v_stages[3].items, &v, 1e-12, "v = B0 u");
        near(&e.v_stages[4].items, &y, 1e-12, "y = W x + b v");
        // Back: c = B0' (b g), dd = c h, db = g v, dx = W' g + A0' (d c).
        let wt = Matrix { rows: 8, cols: 8, items: e.w.items.iter().zip(&e.t.items).map(|(w, t)| w + t).collect() };
        let g: Vec<f64> = y.iter().zip(times(&wt, x)).map(|(y, w)| y - w).collect();
        let c: Vec<f64> = (0..r).map(|k| (0..8).map(|i| e.b0.items[i * r + k] * bv[i] * g[i]).sum()).collect();
        near(&e.v_back_stages[0].items, &g, 1e-12, "g");
        near(&e.v_back_stages[1].items, &c, 1e-12, "c = B0' (b g)");
        near(&e.v_back_stages[2].items, &c.iter().zip(&h).map(|(c, h)| c * h).collect::<Vec<_>>(), 1e-12, "dd = c h");
        near(&e.v_back_stages[3].items, &g.iter().zip(&v).map(|(g, v)| g * v).collect::<Vec<_>>(), 1e-12, "db = g v");
        let dx: Vec<f64> = (0..8).map(|j| (0..8).map(|i| e.w.items[i * 8 + j] * g[i]).sum::<f64>() + (0..r).map(|k| e.a0.items[k * 8 + j] * dv[k] * c[k]).sum::<f64>()).collect();
        near(&e.v_back_stages[4].items, &dx, 1e-12, "dx = W' g + A0' (d c)");
    }
}

#[test]
fn vera_trains_two_vectors_on_shared_projections() {
    for r in [1, 3, 8] {
        let e = at(r);
        assert_eq!(e.trained_v, (8 + r, 16 * r), "rank {r}: VeRA trains 8 + r, LoRA 16 r");
        assert_eq!(e.loss_v[0], e.loss[0], "rank {r}: VeRA starts as the base, b at 0");
        assert!(e.loss_v.last() < e.loss_v.first(), "rank {r}: it learns");
        // The learned update is diag(b) B0 diag(d) A0, and the loss is half
        // its squared distance from T.
        let upd: Vec<f64> = (0..64).map(|t| (0..r).map(|k| e.b_v[t / 8] * e.b0.items[(t / 8) * r + k] * e.d_v[k] * e.a0.items[k * 8 + t % 8]).sum()).collect();
        near(&e.learned_v.items, &upd, 1e-12, "diag(b) B0 diag(d) A0");
        let l: f64 = upd.iter().zip(&e.t.items).map(|(a, t)| (a - t) * (a - t)).sum::<f64>() / 2.0;
        assert!((l - e.loss_v.last().unwrap()).abs() < 1e-9, "rank {r}");
        // Its A0 is LoRA-FA's seeded A: the same seed, shared.
        assert!(*e.loss_v.last().unwrap() >= *e.loss.last().unwrap(), "rank {r}: no better than LoRA");
    }
    // The research's expert, at rank 16: 2 f + d + 3 r, 64 of them, two
    // pairs of projections shared, 2 r (d + f).
    let src = with(Settings::default()).replacen("r := 3\n", "r := 16\n", 1);
    let p = &src[..src.find("### The adapted layer").unwrap()];
    let out = microscope::run::printed(&format!("{p}(r lo:e_xpertV d c_at f) c_at (experts * r lo:e_xpertV d c_at f) c_at 2 * r * d + f\n")).unwrap();
    assert_eq!(out[0], "32816 2100224 589824");
    let e = at(3);
    assert_eq!(e.expert_v, (2 * 14336 + 4096 + 9, 64 * (2 * 14336 + 4096 + 9), 6 * (4096 + 14336)));
}

#[test]
fn the_sweep_s_vera_is_lora_fa_at_rank_1_and_short_of_lora() {
    let w = lora_explorer_web::micro::sweep().unwrap();
    // At rank 1 VeRA's update is (b B0) (d A0's row): any 8-vector times
    // the seeded row, LoRA-FA's at rank 1, which shares the seed.
    assert!((w.vera[0] - w.fa[0]).abs() < 1e-9, "{} against {}", w.vera[0], w.fa[0]);
    for k in 0..8 {
        assert!(w.vera[k] >= w.lora[k] - 1e-9 && w.vera[k] < 1.75, "rank {}: VeRA {}", k + 1, w.vera[k]);
    }
    assert!(program(Settings::default()).contains(lora_explorer_web::micro::backward_v()));
}
