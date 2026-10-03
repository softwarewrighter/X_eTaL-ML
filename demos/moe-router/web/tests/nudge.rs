use moe_router_web::micro::{nudge_core, run_nudge, tables_values, Nudge, EXPERTS, SIDE, SOURCE, STEPS};

/// Scores of the point x0 + a d1 + b d2, computed directly: 2 x . W.
fn scores(e: &[f64], w: &[f64], n: &Nudge, a: f64, b: f64) -> Vec<f64> {
    let row = |i: usize| &e[(i - 1) * 8..i * 8];
    let (x0, xa, xb) = (row(n.w0), row(n.wa), row(n.wb));
    let x: Vec<f64> = (0..8).map(|f| x0[f] + a * (xa[f] - x0[f]) + b * (xb[f] - x0[f])).collect();
    (0..EXPERTS).map(|j| 2.0 * (0..8).map(|f| x[f] * w[f * EXPERTS + j]).sum::<f64>()).collect()
}

/// The first and second expert (1 to 16): softmax keeps the order of
/// the scores, so these are the two largest scores.
fn pair(s: &[f64]) -> (f64, f64) {
    let mut o: Vec<usize> = (0..s.len()).collect();
    o.sort_by(|&i, &j| s[j].total_cmp(&s[i]));
    ((o[0] + 1) as f64, (o[1] + 1) as f64)
}

#[test]
fn the_nudge_is_the_command_line_programs_section() {
    assert!(nudge_core().contains("gs := u:t_op2") && SOURCE.contains(nudge_core()));
}

#[test]
fn boundaries_along_epsilon_match_a_direct_computation() {
    let (e, w) = tables_values().unwrap();
    for n in [Nudge::default(), Nudge { w0: 1, wa: 17, wb: 26 }, Nudge { w0: 23, wa: 30, wb: 12 }] {
        let a = run_nudge(&n).unwrap();
        // Every sample's pair is the direct one.
        for i in 0..STEPS {
            assert_eq!((a.first[i], a.second[i]), pair(&scores(&e, &w, &n, a.eps[i], 0.0)), "{n:?} sample {i}");
        }
        // Exact boundaries, found on a 1000 times finer grid, fall
        // exactly in the intervals where X_eTaL's pair changes.
        let fine = 1000 * (STEPS - 1);
        let mut exact = vec![];
        let mut last = pair(&scores(&e, &w, &n, 0.0, 0.0));
        for i in 1..=fine {
            let p = pair(&scores(&e, &w, &n, 1.2 * i as f64 / fine as f64, 0.0));
            if p != last {
                exact.push((i - 1) / 1000);
                last = p;
            }
        }
        exact.dedup();
        assert_eq!(a.changes(), exact, "{n:?}");
    }
}

#[test]
fn the_slice_matches_a_direct_computation() {
    let (e, w) = tables_values().unwrap();
    let n = Nudge::default();
    let a = run_nudge(&n).unwrap();
    let c = |k: usize| -0.25 + 1.5 * k as f64 / (SIDE - 1) as f64;
    for r in 0..SIDE {
        for col in 0..SIDE {
            let (sa, sb) = (c(col), c(SIDE - 1 - r));
            let i = r * SIDE + col;
            assert_eq!((a.slice_first[i], a.slice_second[i]), pair(&scores(&e, &w, &n, sa, sb)), "point {r} {col}");
        }
    }
}
