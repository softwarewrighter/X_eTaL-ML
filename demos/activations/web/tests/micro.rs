//! The model, natively: every family's curves and slopes as X_eTaL
//! draws them, against the arithmetic done directly; the sliders'
//! parameters; the reader's own activation, drawn, refused, or not
//! finite.

use activations_web::micro::{placed, program, run, run_yours, xs, Params, FAMILIES, POINTS, PROGRAM, YOURS};
use microscope::source::rebound;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn near(got: &[f64], want: impl Fn(f64) -> f64, tol: f64, what: &str) {
    for (x, g) in xs().iter().zip(got) {
        assert!((g - want(*x)).abs() <= tol, "{what} at {x}: {g} != {}", want(*x));
    }
}

#[test]
fn every_family_draws_two_charts_of_its_curves() {
    for (f, fam) in FAMILIES.iter().enumerate() {
        let d = run(f, Params::default()).unwrap_or_else(|e| panic!("{}: {e}", fam.title));
        assert_eq!(d.charts.len(), 2, "{}", fam.title);
        assert!(d.charts.iter().all(|c| c.starts_with("<svg") && c.contains(fam.names[0])), "{}", fam.title);
        assert_eq!((d.ys.len(), d.ds.len()), (fam.names.len(), fam.names.len()));
        assert!(d.ys.iter().chain(&d.ds).all(|c| c.len() == POINTS && c.iter().all(|v| v.is_finite())));
    }
}

#[test]
fn the_curves_are_the_arithmetic_done_directly() {
    let p = Params::default();
    let c = run(0, p).unwrap();
    near(&c.ys[0], sigmoid, 1e-12, "sigmoid");
    near(&c.ys[1], f64::tanh, 1e-12, "tanh");
    near(&c.ys[2], |x| ((x + 3.0).max(0.0).min(6.0)) / 6.0, 1e-12, "hard sigmoid");
    near(&c.ds[0], |x| sigmoid(x) * (1.0 - sigmoid(x)), 1e-12, "sigmoid's slope");
    let r = run(1, p).unwrap();
    near(&r.ys[0], |x| x.max(0.0), 0.0, "ReLU");
    near(&r.ys[1], |x| if x > 0.0 { x } else { p.leak * x }, 1e-15, "leaky ReLU");
    near(&r.ys[2], |x| if x > 0.0 { x } else { p.alpha * (x.exp() - 1.0) }, 1e-12, "ELU");
    let s = run(2, p).unwrap();
    near(&s.ys[0], |x| (1.0 + x.exp()).ln(), 1e-12, "softplus");
    near(&s.ys[1], |x| x * sigmoid(x), 1e-12, "SiLU");
    near(&s.ys[2], |x| x * sigmoid(p.beta * x), 1e-12, "Swish");
    near(&s.ys[3], |x| 0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()), 1e-12, "GELU");
    near(&s.ys[5], |x| x * (1.0 + x.exp()).ln().tanh(), 1e-12, "Mish");
    let g = run(3, p).unwrap();
    near(&g.ys[0], |x| x * sigmoid(x), 1e-12, "GLU, W = V = 1");
    near(&g.ys[1], |x| x * x.max(0.0), 1e-12, "ReGLU");
}

#[test]
fn every_slope_is_its_curves_central_difference_away_from_the_kinks() {
    let h = 1e-4;
    let x = xs();
    for (f, fam) in FAMILIES.iter().enumerate() {
        let d = run(f, Params::default()).unwrap();
        for (c, name) in fam.names.iter().enumerate() {
            for i in 1..POINTS - 1 {
                // The kinks (ReLU's at 0, the hard ones' at -3 and 3) and
                // the slopes there are left out; elsewhere the curve's
                // own differences on the 0.05 grid agree within its curvature.
                if [0.0, -3.0, 3.0].iter().any(|k: &f64| (x[i] - k).abs() < 0.07) {
                    continue;
                }
                let fd = (d.ys[c][i + 1] - d.ys[c][i - 1]) / (x[i + 1] - x[i - 1]);
                assert!((fd - d.ds[c][i]).abs() < 0.02 + 0.01 * d.ds[c][i].abs(), "{name} at {}: {fd} vs {} (h {h})", x[i], d.ds[c][i]);
            }
        }
    }
}

#[test]
fn the_sliders_set_the_parameters() {
    let p = Params { leak: 0.3, alpha: 2.0, beta: 4.0 };
    let r = run(1, p).unwrap();
    assert!((r.ys[1][0] - 0.3 * -6.0).abs() < 1e-12, "leaky ReLU at -6");
    assert!((r.ys[2][0] - 2.0 * ((-6.0f64).exp() - 1.0)).abs() < 1e-12, "ELU at -6");
    let s = run(2, p).unwrap();
    assert!((s.ys[2][POINTS - 1] - 6.0 * sigmoid(24.0)).abs() < 1e-12, "Swish at 6");
    assert!(program(1, p).contains("leak := 0.3\n") && program(1, p).contains("alpha := 2.0\n"));
}

#[test]
fn the_page_shows_the_command_line_programs_code() {
    for f in 0..FAMILIES.len() {
        let src = program(f, Params::default());
        assert!(src.contains(FAMILIES[f].heading));
        // Everything but the read-back lines is the program's own text.
        let body = src.trim_end().rsplitn(3, '\n').nth(2).unwrap();
        assert!(PROGRAM.contains(body.split("### ").nth(1).unwrap()));
        assert!(rebound(&src).is_empty(), "{:?}", rebound(&src));
    }
}

#[test]
fn your_own_activation_is_drawn_beside_a_built_in() {
    let m = run_yours(YOURS, 0).unwrap();
    assert_eq!(m.status, "drawn");
    assert_eq!(m.charts.len(), 2);
    near(&m.ys, |x| x * sigmoid(1.702 * x), 1e-12, "yours");
    near(&m.ds, |x| sigmoid(1.702 * x) + 1.702 * x * sigmoid(1.702 * x) * (1.0 - sigmoid(1.702 * x)), 1e-6, "your slope");
}

#[test]
fn your_own_activation_refused_or_not_finite_says_why() {
    // A mistake, placed in the reader's text (its line 2).
    let e = run_yours("## mine\nu:m_ine := { x -> x * }\n", 0).unwrap_err();
    assert!(e.contains(".xtl:2:"), "{e}");
    // X_eTaL's own refusal (a logarithm of a negative), placed too.
    let e = run_yours("u:m_ine := { x -> l_og x }\n", 0).unwrap_err();
    assert!(e.contains("l_og needs a positive number") && e.contains(".xtl:1:19"), "{e}");
    // A value that is not finite (e^(200 x) overflows past x = 3.55):
    // drawn nothing, the first such x said.
    let m = run_yours("u:m_ine := { x -> e_xp 200.0 * x }\n", 0).unwrap();
    assert!(m.status.starts_with("not finite at x = 3.5"), "{}", m.status);
    assert!(m.charts.is_empty());
}

#[test]
fn an_errors_lines_are_counted_in_the_readers_text() {
    let err = "error[x]: bad\n  --> main.xtl:42:7\n   |\n42 | u:m_ine := { x -> }\n   |";
    assert_eq!(placed(err, 40), "error[x]: bad\n  --> main.xtl:2:7\n   |\n2 | u:m_ine := { x -> }\n   |");
    assert_eq!(placed("  --> main.xtl:3:1", 40), "  --> main.xtl:3:1", "a line before the text is left");
}
