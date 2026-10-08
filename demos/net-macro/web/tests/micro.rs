use net_macro_web::micro::{clean, head, networks, parts, program, run, step_written, train, train_head, train_program, trainable, typed, typed_programs, weight_names, DATA, SIDE, SOURCE, TRAIN_SOURCE, TRAIN_SPEC};

fn data(path: &str) -> Vec<f64> {
    DATA.iter().find(|d| d.0 == path).unwrap().1.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// The network of a spec computed directly in Rust from data/ (loops,
/// no X_eTaL, no macro): the arm decided for (x, y), from 1.
fn direct(spec: &str, weights: &str, x: f64, y: f64) -> usize {
    let mut h = vec![x, y];
    let mut names = weights.split_whitespace();
    for word in spec.split_whitespace().skip(1) {
        match word.parse::<usize>() {
            Ok(n) => {
                let w = data(&format!("data/{}.txt", names.next().unwrap()));
                let m = h.len();
                h = (0..n).map(|j| (0..m).map(|i| h[i] * w[i * n + j]).sum::<f64>() + w[m * n + j]).collect();
            }
            Err(_) => match word {
                "relu" => h.iter_mut().for_each(|v| *v = v.max(0.0)),
                "tanh" => h.iter_mut().for_each(|v| *v = v.tanh()),
                _ => {} // softmax keeps the largest where it is
            },
        }
    }
    1 + (0..h.len()).fold(0, |b, i| if h[i] > h[b] { i } else { b })
}

#[test]
fn the_page_runs_the_command_line_programs_head_and_networks() {
    assert!(SOURCE.contains(head()));
    let n = networks();
    assert_eq!(n.iter().map(|n| n.spec.as_str()).collect::<Vec<_>>(), ["2 3 softmax", "2 4 tanh 3 softmax", "2 16 relu 16 relu 3 softmax"]);
    assert!(n.iter().all(|n| head().contains(&n.line())));
    assert_eq!(n[2].weights(), "c1 c2 c3");
}

#[test]
fn each_macro_written_network_decides_what_the_arithmetic_decides() {
    let pts = data("data/points.txt");
    for n in networks() {
        let r = run(&n).unwrap();
        for row in 0..SIDE {
            for col in 0..SIDE {
                let x = -1.1 + 2.2 * (0.5 + col as f64) / SIDE as f64;
                let y = -1.1 + 2.2 * (0.5 + (SIDE - 1 - row) as f64) / SIDE as f64;
                assert_eq!(r.map[row * SIDE + col], direct(&n.spec, &n.weights(), x, y), "{} at {row} {col}", n.name);
            }
        }
        let right = pts.chunks(3).filter(|p| direct(&n.spec, &n.weights(), p[0], p[1]) == p[2] as usize).count();
        assert!((r.accuracy - right as f64 / (pts.len() / 3) as f64).abs() < 1e-12, "{}", n.name);
    }
}

#[test]
fn the_macro_wrote_the_function_and_counted_the_parameters() {
    let n = networks();
    let deep = run(&n[2]).unwrap();
    assert_eq!(deep.params, 3 * 16 + 17 * 16 + 17 * 3);
    assert!(deep.expansion.contains("u:d_eep := {"), "{}", deep.expansion);
    // The weights' loading, each checked for its count, came from the same spec.
    assert_eq!(deep.expansion.lines().count(), 4, "{}", deep.expansion);
    assert!(deep.expansion.contains("c2 := { v -> 272 = t_ally v ? 17 16 r_eshape v; []P_ANIC"), "{}", deep.expansion);
    assert_eq!(deep.expansion.matches("nn:d_ense").count(), 3);
    assert_eq!(deep.expansion.matches("nn:r_elu").count(), 2);
    assert!(deep.expansion.contains("nn:s_oftmax") && deep.expansion.contains("nn:d_ense c3"));
    assert!(run(&n[0]).unwrap().accuracy < 0.6 && deep.accuracy > 0.95, "a line cannot follow a spiral; the deep network does");
}

#[test]
fn a_typed_spec_is_expanded_and_counted() {
    let t = typed("784 128 relu 10 softmax").unwrap();
    assert_eq!(t.params, 101770);
    assert!(t.expansion.contains("nn:d_ense w1") && t.expansion.contains("nn:d_ense w2") && t.expansion.contains("nn:r_elu"));
    assert_eq!(weight_names("784 128 relu 10 softmax"), "w1 w2");
    assert_eq!(weight_names("3"), "");
}

#[test]
fn a_bad_spec_is_refused_with_the_macros_message() {
    let e = typed("2 8 gelu 3").unwrap_err();
    assert!(e.contains("not a size or an activation") && e.contains("gelu"), "{e}");
    assert!(typed("relu").is_err());
    // A quote cannot break out of the spec's string.
    assert_eq!(clean("2 3\" p_rint! 1 \"softmax"), "2 3  p_rint! 1  softmax");
    assert!(typed("2 3\" p_rint! 1 \"softmax").is_err());
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use net_macro_web::view::program as shown;
    for n in networks() {
        let s = shown(Some(&n), &n.spec);
        for line in program(&n).lines() {
            assert!(s.lines().any(|l| l == line), "not shown: {line}");
        }
    }
    let s = shown(None, "5 4 relu 2");
    let (a, b) = typed_programs("5 4 relu 2");
    for line in a.lines().chain(b.lines()) {
        assert!(s.lines().any(|l| l == line), "not shown: {line}");
    }
}

#[test]
fn the_page_trains_with_train_its_own_lines() {
    let sizes = trainable(TRAIN_SPEC).unwrap();
    // For the program's own spec the page's head is train-it.xtl's.
    assert_eq!(train_head(TRAIN_SPEC, &sizes), TRAIN_SOURCE[TRAIN_SOURCE.find("\"nn:\" u_se<").unwrap()..TRAIN_SOURCE.find("# -- end of the core").unwrap()]);
    let head = train_head("2 8 tanh 3 softmax", &[2, 8, 3]);
    assert!(head.contains("w1 := u:r_andom 3 8\nw2 := u:r_andom 9 3\n") && !head.contains("w3"), "{head}");
    assert!(head.contains("net:t_rain< \"2 8 tanh 3 softmax\"") && head.contains("net:s_tate< \"w1 w2\""), "{head}");
}

#[test]
fn a_spec_trains_here_only_from_2_inputs_to_3_softmax() {
    assert_eq!(trainable("2 16 relu 16 relu 3 softmax").unwrap(), [2, 16, 16, 3]);
    assert_eq!(trainable("2 3 softmax").unwrap(), [2, 3]);
    for bad in ["784 128 relu 10 softmax", "2 4 tanh 3", "2 4 tanh 2 softmax", "2 100 relu 3 softmax", ""] {
        assert!(trainable(bad).is_err(), "{bad}");
    }
    assert_eq!(parts(&[2, 4, 3]), [(3, 4), (5, 3), (3, 4), (5, 3), (3, 4), (5, 3), (1, 1)]);
}

/// The y of each point of a chart's first series (Plot draws it in blue).
fn first_series(svg: &str) -> Vec<f64> {
    let line = svg.split("<polyline").find(|p| p.contains("stroke=\"#2563eb\"")).expect("a first series");
    line.split("points=\"").nth(1).unwrap().split('"').next().unwrap().split(' ').map(|p| p.split(',').nth(1).unwrap().parse().unwrap()).collect()
}

#[test]
fn training_runs_on_from_the_state_the_page_holds() {
    // 40 steps at once equal 20 then 20 carried through the page's state.
    let (spec, sizes) = ("2 8 tanh 3 softmax", vec![2, 8, 3]);
    let start = train(spec, &sizes, None, 0, &[]).unwrap();
    assert!((start.loss - 1.1).abs() < 0.1 && start.pictures.len() == 1, "{} {}", start.loss, start.pictures.len());
    let all = train(spec, &sizes, None, 40, &[]).unwrap();
    let half = train(spec, &sizes, None, 20, &[]).unwrap();
    let rest = train(spec, &sizes, Some(&half.state), 20, &[(start.loss, start.right), (half.loss, half.right)]).unwrap();
    let gap = all.state.iter().zip(&rest.state).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(gap < 1e-9, "{gap}");
    assert!(rest.loss < start.loss - 0.1);
    assert_eq!(rest.points.len(), 900);
    assert_eq!(rest.pictures.len(), 1);
    // Two earlier runs and this one: three points on each line, the loss falling.
    let loss = first_series(&rest.pictures[0]);
    assert!(loss.len() == 3 && loss[2] > loss[0], "{loss:?}");
}

#[test]
fn the_default_spec_learns_the_spiral() {
    let sizes = trainable(TRAIN_SPEC).unwrap();
    let a = train(TRAIN_SPEC, &sizes, None, 200, &[]).unwrap();
    assert!(a.right > 0.9 && a.loss < 0.3, "{} {}", a.loss, a.right);
    let step = step_written(TRAIN_SPEC, &sizes).unwrap();
    assert!(step.starts_with("u:s_tep := {") && step.ends_with('}') && step.contains("nn:r_elu"), "{step}");
    assert!(train_program(TRAIN_SPEC, &sizes, None, 1, &[]).contains("p:c_hart!"));
}
