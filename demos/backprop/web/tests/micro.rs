use backprop_web::micro::{head, program, run, Setup, DATA, I, J, K, N, SOURCE};

fn data(path: &str) -> Vec<f64> {
    DATA.iter().find(|d| d.0 == path).unwrap().1.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

/// The same step computed directly in Rust (loops): the loss and the
/// two gradients for weights w1 (3 x 4) and w2 (5 x 3).
fn direct(w1: &[f64], w2: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let b = data("data/batch.txt");
    let (mut g1, mut g2, mut loss) = (vec![0.0; (I + 1) * J], vec![0.0; (J + 1) * K], 0.0);
    for n in 0..N {
        let x = [b[3 * n], b[3 * n + 1], 1.0];
        let y = b[3 * n + 2] as usize - 1;
        let h: Vec<f64> = (0..J).map(|j| (0..=I).map(|i| x[i] * w1[i * J + j]).sum::<f64>().tanh()).collect();
        let ha: Vec<f64> = h.iter().copied().chain([1.0]).collect();
        let z: Vec<f64> = (0..K).map(|k| (0..=J).map(|j| ha[j] * w2[j * K + k]).sum()).collect();
        let mx = z.iter().cloned().fold(f64::MIN, f64::max);
        let s: f64 = z.iter().map(|v| (v - mx).exp()).sum();
        let p: Vec<f64> = z.iter().map(|v| (v - mx).exp() / s).collect();
        loss -= p[y].ln() / N as f64;
        let d2: Vec<f64> = (0..K).map(|k| (p[k] - (k == y) as u8 as f64) / N as f64).collect();
        for j in 0..=J {
            for k in 0..K {
                g2[j * K + k] += ha[j] * d2[k];
            }
        }
        let d1: Vec<f64> = (0..J).map(|j| (0..K).map(|k| d2[k] * w2[j * K + k]).sum::<f64>() * (1.0 - h[j] * h[j])).collect();
        for i in 0..=I {
            for j in 0..J {
                g1[i * J + j] += x[i] * d1[j];
            }
        }
    }
    (loss, g1, g2)
}

fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn the_page_runs_the_command_line_programs_head() {
    assert!(SOURCE.contains(&head(&Setup::default())));
}

#[test]
fn the_gradients_are_backprop_done_by_hand_and_agree_with_finite_differences() {
    let s = run(&Setup::default()).unwrap();
    let (loss, g1, g2) = direct(&data("data/w1.txt"), &data("data/w2.txt"));
    assert!((s.loss.0 - loss).abs() < 1e-12);
    assert!(close(&s.g1, &g1, 1e-12) && close(&s.g2, &g2, 1e-12));
    assert!(s.gap < 1e-8 && close(&s.f1, &g1, 1e-8) && close(&s.f2, &g2, 1e-8));
    assert!(s.loss.1 < s.loss.0, "one step lowers the loss");
}

#[test]
fn taking_steps_lowers_the_loss_and_too_large_a_rate_does_not() {
    let mut setup = Setup::default();
    let mut losses = vec![];
    for _ in 0..10 {
        let s = run(&setup).unwrap();
        losses.push(s.loss.0);
        let (_, g1, g2) = direct(&s.v1, &s.v2);
        assert!(close(&run(&Setup { weights: Some((s.v1.clone(), s.v2.clone())), ..setup.clone() }).unwrap().g1, &g1, 1e-9));
        let _ = g2;
        setup = Setup { weights: Some((s.v1.clone(), s.v2.clone())), ..setup };
    }
    assert!(losses.windows(2).all(|w| w[1] < w[0]), "{losses:?}");
    let big = run(&Setup { lr: 60.0, ..Setup::default() }).unwrap();
    assert!(big.loss.1 > big.loss.0, "a learning rate of 60 overshoots");
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use backprop_web::view::program as shown;
    let d = Setup::default();
    for line in program(&d).lines() {
        assert!(shown(&d).lines().any(|l| l == line), "not shown: {line}");
    }
    let s = run(&d).unwrap();
    let stepped = Setup { weights: Some((s.v1.clone(), s.v2.clone())), ..d };
    for line in program(&stepped).lines().filter(|l| !l.starts_with("W1 := ") && !l.starts_with("W2 := ")) {
        assert!(shown(&stepped).lines().any(|l| l == line), "not shown: {line}");
    }
    assert!(shown(&stepped).contains("W1 := 3 4 r_eshape ...   # the weights after the steps taken"));
}
