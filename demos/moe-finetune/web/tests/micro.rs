use moe_finetune_web::micro::{compare, compare_program, deploy, deploy_program, files, head, pref_program, prefer, program, run, Adapter, Config, Data, Deploy, Method, Pref, Rate, PROGRAM, STEPS};

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
    // The CLI baseline's 16 rows after the methods': equal ranks, equal budgets, at
    // 0.02, then at 0.1; its last row is VeRA at rank 9 at 0.1.
    let rows: Vec<Vec<f64>> = CLI.lines().skip(14).take(16).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len(), 16);
    let v = compare(Adapter::Vera, Config::EqualBudgets, Rate::Fast, Data::New, 1, STEPS).unwrap();
    assert_eq!(vec![round(v.new), round(v.old), v.numbers as f64], rows[15]);
    // The frozen-projection adapters keep the old task better than
    // expert LoRA at the methods' rate, and learn the new one less.
    for k in [2, 3] {
        assert!(rows[k][1] > rows[1][1] && rows[k][0] < rows[1][0], "row {k}: {:?} against {:?}", rows[k], rows[1]);
    }
}

/// The CLI baseline's deployment lines, after the methods' 14 and the
/// comparison's 16: the merge's 2, the switch's 4, the cache's 6.
fn cli_deploy() -> Vec<Vec<f64>> {
    CLI.lines().skip(30).take(12).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect()
}

#[test]
fn the_deployment_runs_the_command_line_s_lines_and_binds_each_name_once() {
    for part in Deploy::ALL {
        let src = deploy_program(part);
        assert!(microscope::source::rebound(&src).is_empty(), "{part:?}: {:?}", microscope::source::rebound(&src));
        for line in src.lines() {
            assert!(PROGRAM.contains(line), "{part:?}: {line}");
        }
    }
    assert_eq!(cli_deploy().len(), 12);
}

#[test]
fn a_lora_merge_gives_the_same_answers_and_a_qlora_merge_does_not() {
    let d = deploy(Deploy::Merge).unwrap();
    let cli = cli_deploy();
    assert_eq!((d.merge[0].to_vec(), d.merge[1].to_vec()), (cli[0].clone(), cli[1].clone()));
    // LoRA: the probabilities agree to rounding, no answer changes.
    let [gap, changed, kept, merged] = d.merge[0];
    assert!(gap < 1e-12 && changed == 0.0 && kept == merged, "{:?}", d.merge[0]);
    // QLoRA: quantizing W~ + s B A again moves the probabilities and
    // changes some answers.
    let [gap, changed, _, _] = d.merge[1];
    assert!(gap > 0.01 && changed > 0.0, "{:?}", d.merge[1]);
}

#[test]
fn each_task_has_its_adapters_and_the_base_keeps_the_rest() {
    let d = deploy(Deploy::Switch).unwrap();
    let cli = cli_deploy();
    let rows: Vec<Vec<f64>> = d.switch.iter().map(|r| r.to_vec()).collect();
    assert_eq!(rows, cli[2..5].to_vec());
    assert_eq!(d.keeps, cli[5]);
    let (base, one, two) = (d.switch[0], d.switch[1], d.switch[2]);
    // Each task is best served by its own adapters, the rest by the base.
    assert!(one[0] > base[0] && one[0] > two[0], "task 1: {:?}", d.switch);
    assert!(two[1] > base[1] && two[1] > one[1], "task 2: {:?}", d.switch);
    assert!(base[2] > one[2] && base[2] > two[2], "the rest: {:?}", d.switch);
    // Two tasks kept apart: the base once and two adapters.
    assert_eq!(d.keeps, vec![800.0, 384.0, 800.0, 1568.0, 1600.0]);
}

#[test]
fn the_cache_loads_less_with_more_slots_and_kept_apart_moves_less_at_scale() {
    let d = deploy(Deploy::Cache).unwrap();
    let cli = cli_deploy();
    let rows: Vec<Vec<f64>> = d.cache.iter().map(|r| r.to_vec()).collect();
    assert_eq!(rows, cli[6..10].to_vec());
    assert_eq!((&d.scale, &d.scale_cache), (&cli[10], &cli[11]));
    for w in d.cache.windows(2) {
        assert!(w[1][1] <= w[0][1] && w[1][2] <= w[0][2], "{w:?}");
    }
    // 50 batches: every one loads something at 1 slot; 4 slots hold the
    // 4 base experts, loaded once each.
    assert_eq!(d.cache[3][1], 4.0);
    // Merged copies are keys of their own per task, so they load as often
    // as the adapters; with VeRA kept apart, far fewer bytes move.
    for r in &d.cache {
        assert!(r[6] < r[5] && r[5] < r[4], "VeRA < LoRA-FA < LoRA: {r:?}");
    }
    // At the research's size an adapter is small beside its expert, so
    // keeping it apart moves about half the merged copies' bytes.
    let s = &d.scale_cache;
    assert!(s[4] < 0.6 * s[3] && s[6] < s[4], "{s:?}");
}

/// The CLI baseline's preference rows, after the deployment's 12.
fn cli_pref() -> Vec<Vec<f64>> {
    CLI.lines().skip(42).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect()
}

#[test]
fn preference_runs_bind_each_name_once_and_are_the_command_line_s() {
    let cli = cli_pref();
    assert_eq!(cli.len(), 6);
    for (p, row) in Pref::ALL.into_iter().zip(&cli) {
        let src = pref_program(p, 1, STEPS);
        assert!(microscope::source::rebound(&src).is_empty(), "{p:?}: {:?}", microscope::source::rebound(&src));
        let r = prefer(p, 1, STEPS).unwrap();
        assert_eq!(vec![r.valid, r.preferred, r.old], *row, "{p:?}");
    }
}

#[test]
fn sft_learns_what_is_right_and_dpo_on_valid_pairs_what_is_better() {
    let r: Vec<Vec<f64>> = cli_pref();
    let (base, sft_preferred, sft_valid, dpo1, dpo5, checked) = (&r[0], &r[1], &r[2], &r[3], &r[4], &r[5]);
    // SFT on either valid answer: valid nearly always, preferred little
    // more than half the time.
    assert!(sft_valid[0] > 0.9 && sft_valid[1] < 0.65, "{sft_valid:?}");
    // DPO on the valid pairs learns the preference, past SFT on the
    // preferred answers alone, and stays valid.
    for d in [dpo1, dpo5] {
        assert!(d[1] > sft_valid[1] + 0.2 && d[1] > sft_preferred[1] && d[0] >= sft_valid[0], "{d:?}");
    }
    // Pairs of right against wrong raise validity, not the preference.
    assert!(checked[0] > sft_valid[0] && checked[1] < dpo1[1] - 0.15, "{checked:?}");
    // A larger beta keeps closer to the reference: less of the old task lost.
    assert!(dpo5[2] > dpo1[2], "{dpo5:?} {dpo1:?}");
    assert!(base[2] > sft_valid[2], "fine-tuning shared experts still costs the old task");
}

#[test]
fn dpo_s_gradients_are_the_finite_differences() {
    // A made-up adapter on every expert, against the base as reference;
    // the DPO loss's gradient for the experts' A and B against central
    // differences of the loss g_radsD gives.
    let src = format!(
        "{}tA := 5 2 9 r_eshape 0.3 * s_in 0.7 * f_loat (r_ange 90) * r_ange 90\n\
         tB := 5 3 2 r_eshape 0.3 * s_in 1.1 * f_loat (r_ange 30) * r_ange 30\n\
         (tS, tbS, tR, tW, _, _) := base\n\
         (_, _, tP) := base u:f_orward6 x\n\
         u:l_d := {{ (A, B) ->\n  (_, _, _, _, _, _, L) := (tS, tbS, tR, tW, A, B) u:g_radsD (styled, tP, 2.0)\n  L\n}}\n\
         (_, _, _, _, gA, gB, _) := (tS, tbS, tR, tW, tA, tB) u:g_radsD (styled, tP, 2.0)\n\
         h := 0.00001\n\
         u:b_ump := {{ w i -> h * f_loat (s_hape w) r_eshape i = r_ange t_ally r_avel w }}\n\
         fA := '{{ i -> ((u:l_d (tA + tA u:b_ump i, tB)) - u:l_d (tA - tA u:b_ump i, tB)) / 2.0 * h }} e_ach r_ange 72\n\
         fB := '{{ i -> ((u:l_d (tA, tB + tB u:b_ump i)) - u:l_d (tA, tB - tB u:b_ump i)) / 2.0 * h }} e_ach r_ange 24\n\
         ('m_ax r_/ a_bs fA - 72 t_ake r_avel gA) c_at 'm_ax r_/ a_bs fB - 24 t_ake r_avel gB\n",
        head()
    );
    let out = microscope::run::output_with(&files(), &src, 1).unwrap().0;
    let gaps: Vec<f64> = out[0].split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert!(gaps.iter().all(|&g| g < 1e-6), "{gaps:?}");
}
