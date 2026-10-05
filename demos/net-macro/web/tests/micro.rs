use net_macro_web::micro::{clean, head, networks, program, run, typed, typed_programs, weight_names, DATA, SIDE, SOURCE};

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
    assert!(n.iter().all(|n| head().contains(&format!("u:{} := \"{}\" net:n_etwork< \"{}\"", n.name, n.spec, n.weights))));
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
                assert_eq!(r.map[row * SIDE + col], direct(&n.spec, &n.weights, x, y), "{} at {row} {col}", n.name);
            }
        }
        let right = pts.chunks(3).filter(|p| direct(&n.spec, &n.weights, p[0], p[1]) == p[2] as usize).count();
        assert!((r.accuracy - right as f64 / (pts.len() / 3) as f64).abs() < 1e-12, "{}", n.name);
    }
}

#[test]
fn the_macro_wrote_the_function_and_counted_the_parameters() {
    let n = networks();
    let deep = run(&n[2]).unwrap();
    assert_eq!(deep.params, 3 * 16 + 17 * 16 + 17 * 3);
    assert!(deep.expansion.starts_with("u:d_eep := ("), "{}", deep.expansion);
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
