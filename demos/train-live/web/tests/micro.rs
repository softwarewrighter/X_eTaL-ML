use train_live_web::micro::{head, program, run, POINTS, SIDE, SOURCE, STATE};

/// The same training computed directly in Rust from X_eTaL's own
/// points and starting state: the loss after `steps` Adam steps.
fn direct(x: &[f64], arms: &[usize], s0: &[f64], steps: usize, lr: f64) -> (Vec<f64>, f64) {
    let (n, hdim) = (arms.len(), 16);
    let loss_grad = |w: &[f64]| -> (f64, Vec<f64>) {
        let (w1, w2) = (&w[..48], &w[48..99]);
        let mut g = vec![0.0; 99];
        let mut loss = 0.0;
        for p in 0..n {
            let xi = [x[2 * p], x[2 * p + 1], 1.0];
            let h: Vec<f64> = (0..hdim).map(|j| (0..3).map(|i| xi[i] * w1[i * hdim + j]).sum::<f64>().tanh()).collect();
            let ha: Vec<f64> = h.iter().copied().chain([1.0]).collect();
            let z: Vec<f64> = (0..3).map(|k| (0..=hdim).map(|j| ha[j] * w2[j * 3 + k]).sum()).collect();
            let mx = z.iter().cloned().fold(f64::MIN, f64::max);
            let s: f64 = z.iter().map(|v| (v - mx).exp()).sum();
            let pr: Vec<f64> = z.iter().map(|v| (v - mx).exp() / s).collect();
            let y = arms[p] - 1;
            loss -= pr[y].ln() / n as f64;
            let d2: Vec<f64> = (0..3).map(|k| (pr[k] - (k == y) as u8 as f64) / n as f64).collect();
            for j in 0..=hdim {
                for k in 0..3 {
                    g[48 + j * 3 + k] += ha[j] * d2[k];
                }
            }
            for j in 0..hdim {
                let d1 = (0..3).map(|k| d2[k] * w2[j * 3 + k]).sum::<f64>() * (1.0 - h[j] * h[j]);
                for i in 0..3 {
                    g[i * hdim + j] += xi[i] * d1;
                }
            }
        }
        (loss, g)
    };
    let mut s = s0.to_vec();
    for _ in 0..steps {
        let k = s[297] + 1.0;
        let (_, g) = loss_grad(&s[..99]);
        for i in 0..99 {
            let m = 0.9 * s[99 + i] + 0.1 * g[i];
            let v = 0.999 * s[198 + i] + 0.001 * g[i] * g[i];
            s[i] -= lr * (m / (1.0 - 0.9f64.powf(k))) / (1e-8 + (v / (1.0 - 0.999f64.powf(k))).sqrt());
            s[99 + i] = m;
            s[198 + i] = v;
        }
        s[297] = k;
    }
    let (loss, _) = loss_grad(&s[..99]);
    (s, loss)
}

#[test]
fn the_page_runs_the_command_line_programs_head() {
    assert!(SOURCE.contains(&head(0.02)));
}

#[test]
fn x_etal_trains_as_adam_in_rust_does_and_learns_the_spiral() {
    let a0 = run(0.02, None, 0).unwrap();
    assert_eq!(a0.state.len(), STATE);
    assert_eq!((a0.points.len(), a0.arms.len(), a0.map.len()), (2 * POINTS, POINTS, SIDE * SIDE));
    let (s50, l50) = direct(&a0.points, &a0.arms, &a0.state, 50, 0.02);
    let a50 = run(0.02, Some(&a0.state), 50).unwrap();
    assert!(a50.state.iter().zip(&s50).all(|(p, q)| (p - q).abs() < 1e-9), "50 Adam steps agree");
    assert!((a50.loss - l50).abs() < 1e-9);
    // Run by run, as the page does, to 400 steps.
    let mut s = a50.state.clone();
    let mut last = a50.loss;
    for _ in 0..14 {
        let a = run(0.02, Some(&s), 25).unwrap();
        assert!(a.loss < last + 0.05, "the loss keeps falling: {} after {last}", a.loss);
        last = a.loss;
        s = a.state;
    }
    let a = run(0.02, Some(&s), 0).unwrap();
    assert!(a.right > 0.95 && a.loss < 0.2, "after 400 steps: {} right, loss {}", a.right, a.loss);
    assert!(a0.right < 0.6, "untrained: {}", a0.right);
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use train_live_web::view::program as shown;
    let s = shown(0.02, true, 25);
    for line in program(0.02, Some(&[0.5; STATE]), 25).lines().filter(|l| !l.starts_with("s := 298 r_eshape ")) {
        assert!(s.lines().any(|l| l == line), "not shown: {line}");
    }
    let s0 = shown(0.02, false, 25);
    for line in program(0.02, None, 25).lines() {
        assert!(s0.lines().any(|l| l == line), "not shown: {line}");
    }
}
