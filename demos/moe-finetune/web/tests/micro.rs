use moe_finetune_web::micro::{compare, compare_program, files, head, program, run, Adapter, Config, Data, Method, Rate, PROGRAM, STEPS};

/// The command line's baseline: the mixed tokens' rows at seed 1.
const CLI: &str = include_str!("../../reg/cli-moe-finetune.out");

#[test]
fn the_page_runs_the_programs_core_and_binds_each_name_once() {
    assert!(PROGRAM.contains(head()));
    for m in Method::ALL {
        for d in [Data::New, Data::Mixed] {
            let src = program(m, d, 2, STEPS);
            assert!(microscope::source::rebound(&src).is_empty(), "{m:?}: {:?}", microscope::source::rebound(&src));
        }
    }
}

#[test]
fn the_base_and_its_room_for_a_new_expert() {
    // No method: the base, the fifth expert never chosen.
    let r = run(Method::None, Data::Mixed, 1, STEPS).unwrap();
    assert_eq!((r.new, r.old, r.numbers, r.losses.len()), (0.485, 0.91, 0, 0));
    assert_eq!((r.share_new[4], r.share_old[4]), (0.0, 0.0));
    assert_eq!(r.share_new, r.base_new);
    for share in [&r.base_new, &r.base_old] {
        assert!((share.iter().sum::<f64>() - 2.0).abs() < 1e-9, "each token keeps two experts");
    }
}

#[test]
fn what_each_method_trains_and_what_it_leaves() {
    // A few steps each: the counts, and which updates move.
    let want = [(Method::Shared, 34), (Method::Experts, 96), (Method::Both, 130), (Method::Selected, 48), (Method::Router, 132), (Method::Added, 36)];
    for (m, n) in want {
        let r = run(m, Data::New, 1, 3).unwrap();
        assert_eq!((r.numbers, r.bytes, r.losses.len()), (n, 4 * n, 3), "{m:?}");
        let moved: Vec<bool> = r.sizes.iter().map(|&s| s > 0.0).collect();
        let shared = matches!(m, Method::Shared | Method::Both);
        let experts = matches!(m, Method::Experts | Method::Both | Method::Router);
        assert_eq!(moved[4], shared, "{m:?}: the shared layer's adapter");
        assert_eq!(moved[5], m == Method::Added, "{m:?}: the new expert");
        if experts {
            assert!(moved[..4].iter().all(|&b| b), "{m:?}: every expert's adapter");
        }
        if m == Method::Selected {
            assert_eq!(moved[..4].iter().filter(|&&b| b).count(), 2, "two experts, the new tokens' likeliest");
        }
        // With the router frozen and the shared layer as it was, the
        // tokens go where they went.
        if matches!(m, Method::Experts | Method::Selected) {
            assert_eq!((&r.share_new, &r.share_old), (&r.base_new, &r.base_old), "{m:?}");
        }
    }
    // The new expert's row starts pointing at the new tokens.
    let r = run(Method::Added, Data::New, 1, 0).unwrap();
    assert!(r.share_new[4] > 0.8 && r.share_old[4] < 0.1, "{:?} {:?}", r.share_new, r.share_old);
}

#[test]
fn a_run_is_the_command_lines_and_only_the_new_expert_keeps_the_old_task() {
    // The CLI baseline's last 7 rows: every method on the mixed tokens,
    // seed 1: the new task, the old one, the numbers.
    let rows: Vec<Vec<f64>> = CLI.lines().skip(7).take(7).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len(), 7);
    let base = rows[0][1];
    for (i, row) in rows.iter().enumerate() {
        let kept = base - row[1] <= 0.03;
        assert_eq!(kept, i == 0 || i == 6, "row {i}: {row:?}");
    }
    // The page's run of one method is the command line's row.
    let r = run(Method::Added, Data::Mixed, 1, STEPS).unwrap();
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    assert_eq!((round(r.new), round(r.old), r.numbers as f64), (rows[6][0], rows[6][1], rows[6][2]));
}

#[test]
fn the_comparison_binds_each_name_once_and_counts_what_each_adapter_trains() {
    for a in Adapter::ALL {
        for c in Config::ALL {
            let src = compare_program(a, c, Rate::Fast, Data::New, 1, STEPS);
            assert!(microscope::source::rebound(&src).is_empty(), "{a:?} {c:?}: {:?}", microscope::source::rebound(&src));
        }
    }
    // r (8 + 9), 4 r (3 + 9), 4 x 3 r, 4 (3 + r): equal ranks, then about 48.
    let want = [(Config::EqualRanks, [34, 96, 24, 20]), (Config::EqualBudgets, [51, 48, 48, 48])];
    for (c, ns) in want {
        for (a, n) in Adapter::ALL.into_iter().zip(ns) {
            let r = compare(a, c, Rate::Methods, Data::New, 1, 1).unwrap();
            assert_eq!((r.numbers, r.losses.len()), (n, 1), "{a:?} {c:?}");
        }
    }
}

#[test]
fn each_kind_s_gradients_are_the_finite_differences() {
    // The model of each kind from made-up P and Q, its gradients chained
    // back to them, against central differences of the loss.
    let core = head();
    let check = |kind: usize, r: usize, p: &str, q: &str| {
        let src = format!(
            "{core}(tA, tB) := 2 lo:p_rojections 3 9 c_at {r}\n\
             tc := ({kind}, {r}, tA, tB)\n\
             tP := {p}\ntQ := {q}\n\
             u:l_t := {{ (P, Q) ->\n  (_, _, _, _, _, _, L) := (tc u:e_ffective (P, Q)) u:g_rads6 (x, Y)\n  L\n}}\n\
             (gS, gB, _, _, gA, gE, _) := (tc u:e_ffective (tP, tQ)) u:g_rads6 (x, Y)\n\
             (dP, dQ) := tc u:c_hain (gS, gB, gA, gE)\n\
             h := 0.00001\n\
             u:b_ump := {{ w i -> h * f_loat (s_hape w) r_eshape i = r_ange t_ally r_avel w }}\n\
             fP := '{{ i -> ((u:l_t (tP + tP u:b_ump i, tQ)) - u:l_t (tP - tP u:b_ump i, tQ)) / 2.0 * h }} e_ach r_ange t_ally r_avel tP\n\
             fQ := '{{ i -> ((u:l_t (tP, tQ + tQ u:b_ump i)) - u:l_t (tP, tQ - tQ u:b_ump i)) / 2.0 * h }} e_ach r_ange t_ally r_avel tQ\n\
             ('m_ax r_/ a_bs fP - r_avel dP) c_at 'm_ax r_/ a_bs fQ - r_avel dQ\n"
        );
        let out = microscope::run::output_with(&files(), &src, 1).unwrap_or_else(|e| panic!("kind {kind}: {e}")).0;
        let gaps: Vec<f64> = out[0].split_whitespace().map(|v| v.parse().unwrap()).collect();
        gaps
    };
    let ripple = |n: usize, c: f64| format!("0.3 * s_in {c} * f_loat (r_ange {n}) * r_ange {n}");
    // Shared LoRA at rank 2; expert LoRA at 2; VeRA at 3 (b away from 0).
    let shared = check(1, 2, &format!("2 9 r_eshape {}", ripple(18, 0.7)), &format!("8 2 r_eshape {}", ripple(16, 1.1)));
    assert!(shared.iter().all(|&g| g < 1e-6), "shared LoRA: {shared:?}");
    let expert = check(2, 2, &format!("4 2 9 r_eshape {}", ripple(72, 0.7)), &format!("4 3 2 r_eshape {}", ripple(24, 1.1)));
    assert!(expert.iter().all(|&g| g < 1e-6), "expert LoRA: {expert:?}");
    let vera = check(4, 3, &format!("4 3 r_eshape 0.5 + {}", ripple(12, 0.7)), &format!("4 3 r_eshape {}", ripple(12, 1.1)));
    assert!(vera.iter().all(|&g| g < 1e-6), "VeRA: {vera:?}");
    // LoRA-FA: B's gradient is LoRA's; A's is 0, so its difference is A's whole gradient.
    let fa = check(3, 2, &format!("4 2 9 r_eshape {}", ripple(72, 0.7)), &format!("4 3 2 r_eshape {}", ripple(24, 1.1)));
    assert!(fa[1] < 1e-6 && fa[0] > 1e-4, "LoRA-FA: {fa:?}");
}

#[test]
fn the_comparison_s_runs_are_the_methods_and_the_command_line_s() {
    // At rank 2 shared LoRA is the method "shared layer only" and expert
    // LoRA "every expert, router frozen": the same start, the same steps.
    let round = |v: f64| (v * 1000.0).round() / 1000.0;
    let s = compare(Adapter::Shared, Config::EqualRanks, Rate::Methods, Data::New, 1, STEPS).unwrap();
    let m = run(Method::Shared, Data::New, 1, STEPS).unwrap();
    assert_eq!((s.new, s.old, &s.losses), (m.new, m.old, &m.losses));
    let e = compare(Adapter::Expert, Config::EqualRanks, Rate::Methods, Data::New, 1, STEPS).unwrap();
    let me = run(Method::Experts, Data::New, 1, STEPS).unwrap();
    assert_eq!((e.new, e.old), (me.new, me.old));
    // The CLI baseline's last 16 rows: equal ranks, equal budgets, at
    // 0.02, then at 0.1; its last row is VeRA at rank 9 at 0.1.
    let rows: Vec<Vec<f64>> = CLI.lines().skip(14).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len(), 16);
    let v = compare(Adapter::Vera, Config::EqualBudgets, Rate::Fast, Data::New, 1, STEPS).unwrap();
    assert_eq!(vec![round(v.new), round(v.old), v.numbers as f64], rows[15]);
    // The frozen-projection adapters keep the old task better than
    // expert LoRA at the methods' rate, and learn the new one less.
    for k in [2, 3] {
        assert!(rows[k][1] > rows[1][1] && rows[k][0] < rows[1][0], "row {k}: {:?} against {:?}", rows[k], rows[1]);
    }
}
