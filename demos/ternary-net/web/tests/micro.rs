use microscope::run::{numbers, output};
use ternary_net_web::micro::{core, run, weights, Setup, SETS, SIDE, SOURCE};

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn the_page_runs_the_command_line_programs_sections() {
    assert!(weights().contains("qa := ") && weights().contains("pl := "));
    assert!(core().contains("u:n_et := { x k ->") && core().contains("u:t_ern := { k x ->"));
    assert!(SOURCE.contains(weights()) && SOURCE.contains(core()));
}

#[test]
fn training_for_ternary_is_what_makes_ternary_work() {
    let fp = run(&Setup::default()).unwrap();
    let qa = run(&Setup { set: 1, ..Setup::default() }).unwrap();
    assert_eq!((SETS[0].1, SETS[1].1), ("fp", "qa"));
    assert_eq!(fp.measures[0][0], 1.0, "the full-precision net gets every test point");
    assert_eq!(qa.measures[3][0], 1.0, "the net trained for ternary gets every test point as ternary");
    assert!(fp.measures[3][0] < 0.9, "the full-precision net, ternarized, loses points: {}", fp.measures[3][0]);
}

#[test]
fn fp32_is_the_reference_and_coarser_formats_drift_more() {
    let a = run(&Setup::default()).unwrap();
    assert_eq!(a.measures[0][1], 1.0);
    assert_eq!(a.measures[0][2], 0.0);
    let (e16, e8, e2) = (a.measures[1][2], a.measures[2][2], a.measures[3][2]);
    assert!(e16 < e8 && e8 < e2, "errors {e16} {e8} {e2}");
    assert!(e16 < 0.01);
}

#[test]
fn ternary_weights_are_minus_one_zero_or_one() {
    let a = run(&Setup::default()).unwrap();
    assert!(a.q.iter().all(|&v| v == -1.0 || v == 0.0 || v == 1.0));
    // Padding stays 0: layer 1 has 2 inputs, layer 3 has 3 outputs.
    for i in 2..16 {
        assert!((0..16).all(|o| a.q(i, 0, o) == 0.0));
    }
    assert!((0..16).all(|i| (3..16).all(|o| a.q(i, 2, o) == 0.0)));
    assert!(a.scales.iter().all(|&s| s > 0.0));
}

#[test]
fn a_higher_threshold_keeps_fewer_weights() {
    let kept: Vec<usize> = [0.0, 0.5, 1.0, 1.5].iter().map(|&t| run(&Setup { t, ..Setup::default() }).unwrap().kept()).collect();
    assert!(kept.windows(2).all(|w| w[0] > w[1]), "{kept:?}");
    assert!(kept[0] > 320, "threshold 0 keeps (nearly) all 336 weights: {}", kept[0]);
}

#[test]
fn the_ternary_layer_is_the_masked_sum_formulation() {
    for point in [(0.3, 0.4), (-0.7, 0.1), (0.05, -0.9)] {
        let a = run(&Setup { point, ..Setup::default() }).unwrap();
        for o in 0..16 {
            // Additions and subtractions of the inputs, as the page lists them.
            let plus: f64 = (0..16).filter(|&i| a.q(i, 1, o) == 1.0).map(|i| a.h1[i]).sum();
            let minus: f64 = (0..16).filter(|&i| a.q(i, 1, o) == -1.0).map(|i| a.h1[i]).sum();
            assert!(close(a.adds[o], plus - minus, 1e-12));
            // One multiply by the scale, plus the bias: the layer's output.
            assert!(close(a.scales[1] * a.adds[o] + a.b2[o], a.pre[o], 1e-9), "{point:?} unit {o}");
        }
    }
}

#[test]
fn the_network_agrees_with_a_direct_forward_pass() {
    // The model as X_eTaL holds it, then a forward pass in Rust.
    let src = format!("{}r_avel fp\n", weights());
    let m = numbers::<f64>(&output(&src, 1).unwrap()[0], 17 * 3 * 16).unwrap();
    let w = |r: usize, l: usize, o: usize| m[r * 48 + l * 16 + o];
    let (x, y) = (0.3, 0.4);
    let mut h: Vec<f64> = vec![x, y];
    for (l, (ni, no)) in [(2, 16), (16, 16), (16, 3)].into_iter().enumerate() {
        let mut z: Vec<f64> = (0..no).map(|o| w(16, l, o)).collect();
        for (o, zo) in z.iter_mut().enumerate() {
            for i in 0..ni {
                *zo += h[i] * w(i, l, o);
            }
        }
        h = if l < 2 { z.iter().map(|v| v.max(0.0)).collect() } else { z };
    }
    let a = run(&Setup { point: (x, y), ..Setup::default() }).unwrap();
    for o in 0..3 {
        // FP32 rounds each weight to 24 bits: agreement to about 1e-6.
        assert!(close(a.outs[0][o], h[o], 1e-5), "output {o}: {} vs {}", a.outs[0][o], h[o]);
    }
    assert_eq!(a.maps[0].len(), SIDE * SIDE);
}
