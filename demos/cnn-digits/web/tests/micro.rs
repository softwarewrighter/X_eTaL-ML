use cnn_digits_web::micro::{head, program, run, Anatomy, Input, DATA, SOURCE};

fn data(path: &str) -> Vec<f64> {
    let text = DATA.iter().find(|d| d.0 == path).unwrap().1;
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// The same network computed directly in Rust from data/ (loops, no
/// X_eTaL): convolution and ReLU, pooling, dense, softmax.
fn direct(x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fb = data("data/filters.txt");
    let dense = data("data/dense.txt");
    let mut c = vec![0.0; 8 * 26 * 26];
    for f in 0..8 {
        for r in 0..26 {
            for k in 0..26 {
                let mut s = fb[10 * f + 9];
                for a in 0..3 {
                    for b in 0..3 {
                        s += fb[10 * f + 3 * a + b] * x[(r + a) * 28 + k + b];
                    }
                }
                c[(f * 26 + r) * 26 + k] = s.max(0.0);
            }
        }
    }
    let mut m = vec![0.0; 8 * 13 * 13];
    for f in 0..8 {
        for r in 0..13 {
            for k in 0..13 {
                let at = |i: usize, j: usize| c[(f * 26 + 2 * r + i) * 26 + 2 * k + j];
                m[(f * 13 + r) * 13 + k] = at(0, 0).max(at(0, 1)).max(at(1, 0)).max(at(1, 1));
            }
        }
    }
    let mut z: Vec<f64> = dense[1352 * 10..].to_vec();
    for (i, &v) in m.iter().enumerate() {
        for o in 0..10 {
            z[o] += v * dense[i * 10 + o];
        }
    }
    let mx = z.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - mx).exp()).collect();
    let s: f64 = e.iter().sum();
    (c, m, e.iter().map(|v| v / s).collect())
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn the_page_runs_the_command_line_programs_head() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:c_onv := ") && head().contains("n_umbers []N_GET \"data/dense.txt\""));
    assert!(program(&Input::Sample(7)).starts_with(head()));
}

#[test]
fn x_etal_computes_the_same_network_as_rust_for_every_sample() {
    let samples = data("data/samples.txt");
    let expected = data("data/expected.txt");
    for d in 0..10 {
        let a: Anatomy = run(&Input::Sample(d)).unwrap();
        assert_eq!(a.x, samples[784 * d..784 * (d + 1)].to_vec());
        let (c, m, p) = direct(&a.x);
        assert!(close(&a.c, &c, 1e-9), "conv of {d}");
        assert!(close(&a.m, &m, 1e-9), "pool of {d}");
        assert!(close(&a.p, &p, 1e-9), "probabilities of {d}");
        assert!(close(&a.p, &expected[10 * d..10 * (d + 1)], 1e-9), "trainer's probabilities of {d}");
        assert_eq!(a.digit(), d, "the network reads sample {d} right");
    }
}

#[test]
fn the_patch_arithmetic_is_x_etals_value() {
    let a = run(&Input::Sample(7)).unwrap();
    for (f, r, c) in [(0, 0, 0), (2, 12, 12), (5, 20, 7), (7, 25, 25)] {
        let p = a.patch(f, r, c);
        assert!((p.value - a.c_at(f, r, c)).abs() < 1e-9, "({f}, {r}, {c})");
    }
}

#[test]
fn the_strongest_answer_is_the_maps_largest() {
    let a = run(&Input::Sample(7)).unwrap();
    for f in 0..8 {
        let (r, c) = a.strongest(f);
        let top = a.c[f * 676..(f + 1) * 676].iter().cloned().fold(f64::MIN, f64::max);
        assert_eq!(a.c_at(f, r, c), top);
    }
}

#[test]
fn a_drawn_digit_runs_like_the_same_sample() {
    let a = run(&Input::Sample(3)).unwrap();
    let b = run(&Input::Drawn(a.x.clone())).unwrap();
    assert!(close(&a.p, &b.p, 1e-12));
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use cnn_digits_web::view::program as shown;
    for input in [Input::Sample(7), Input::Drawn(vec![0.5; 784])] {
        let s = shown(&input);
        for line in program(&input).lines().filter(|l| !l.starts_with("x := 28 28 r_eshape ")) {
            assert!(s.lines().any(|l| l == line), "not shown: {line}");
        }
    }
    assert!(shown(&Input::Drawn(vec![0.0; 784])).contains("x := 28 28 r_eshape ...   # the digit you drew, 784 numbers"));
}

#[test]
fn a_stroke_is_unbroken() {
    use cnn_digits_web::model::stroke;
    let mut pad = vec![0.0; 784];
    stroke(&mut pad, (2, 14), (25, 14));
    assert!((2..=25).all(|r| pad[r * 28 + 14] == 1.0), "every cell on the line is inked");
}
