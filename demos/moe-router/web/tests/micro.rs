use moe_router_web::micro::{core, run, tables, tokens, words, Anatomy, EXPERTS, SENTENCES, SOURCE};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn routed() -> Vec<Anatomy> {
    SENTENCES.iter().map(|s| run(s).unwrap_or_else(|e| panic!("{s}: {e}"))).collect()
}

#[test]
fn the_page_runs_the_command_line_programs_sections() {
    assert!(tables().contains("W := ") && core().contains("u:t_op2 := { p ->"));
    assert!(SOURCE.contains(tables()) && SOURCE.contains(core()));
    assert_eq!(words().len(), 37);
    assert_eq!(*words().last().unwrap(), "?");
}

#[test]
fn words_are_looked_up_with_plurals_and_unknowns() {
    let t = tokens("Two brown Horses swim.");
    let w = words();
    assert_eq!(t.iter().map(|(_, i)| w[i - 1]).collect::<Vec<_>>(), ["two", "brown", "horse", "?"]);
}

#[test]
fn probabilities_sum_to_one_and_follow_the_scores() {
    for a in routed() {
        for t in 0..a.n() {
            let r = t * EXPERTS..(t + 1) * EXPERTS;
            assert!(close(a.p[r.clone()].iter().sum(), 1.0));
            // softmax: p_e / p_f = exp(s_e - s_f)
            let (s, p) = (&a.scores[r.clone()], &a.p[r]);
            assert!(close(p[3] / p[7], (s[3] - s[7]).exp()));
        }
    }
}

#[test]
fn top2_is_the_two_largest_by_a_direct_sort_and_gates_sum_to_one() {
    for a in routed() {
        for t in 0..a.n() {
            let p = &a.p[t * EXPERTS..(t + 1) * EXPERTS];
            let mut order: Vec<usize> = (0..EXPERTS).collect();
            order.sort_by(|&x, &y| p[y].total_cmp(&p[x]));
            assert_eq!(a.top2(t), [order[0], order[1]], "token {t} of {:?}", a.words);
            let g = &a.gates[t * EXPERTS..(t + 1) * EXPERTS];
            assert!(close(g.iter().sum(), 1.0));
            assert!(close(g[order[0]], p[order[0]] / (p[order[0]] + p[order[1]])));
            assert_eq!(g.iter().filter(|&&v| v > 0.0).count(), 2);
        }
    }
}

#[test]
fn the_load_counts_each_experts_tokens() {
    for a in routed() {
        for e in 0..EXPERTS {
            let count = (0..a.n()).filter(|&t| a.top2(t).contains(&e)).count();
            assert_eq!(a.load[e], count as f64);
        }
        assert_eq!(a.load.iter().sum::<f64>(), 2.0 * a.n() as f64);
    }
}

#[test]
fn experts_specialise_by_feature() {
    let a = run("the red fox eats fish in the park").unwrap();
    // red: a color expert (grid row 3); fish: animal x food (expert 2);
    // park: a place expert (grid column 1).
    assert_eq!(a.top2(1)[0] / 4, 2);
    assert_eq!(a.top2(4)[0], 1);
    assert_eq!(a.top2(7)[0] % 4, 0);
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use moe_router_web::micro::{nudge_program, program as sentence_program, Nudge};
    use moe_router_web::view::program;
    let ids = [1, 17, 9, 23];
    let n = Nudge::default();
    let shown = program(&ids, &n);
    for p in [sentence_program(&ids), nudge_program(&n)] {
        for line in p.lines() {
            assert!(shown.lines().any(|l| l == line), "not shown: {line}");
        }
    }
    assert!(shown.contains("F := 37 8 r_eshape n_umbers []N_GET \"data/features.txt\""));
}
