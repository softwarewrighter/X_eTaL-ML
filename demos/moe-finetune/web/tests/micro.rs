use moe_finetune_web::micro::{head, program, run, Data, Method, PROGRAM, STEPS};

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
    let rows: Vec<Vec<f64>> = CLI.lines().skip(7).map(|l| l.split_whitespace().map(|v| v.parse().unwrap()).collect()).collect();
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
