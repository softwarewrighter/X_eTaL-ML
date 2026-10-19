//! The page's model, natively: each example runs, agrees, and shows
//! code taken from the command-line programs; the CNN layer's maps are
//! checked against the arithmetic done directly in Rust.

use microscope::source::rebound;
use pipelines_web::micro::{calls, head, program, run, statement, EXAMPLES, PIPELINES, ROSETTA};

#[test]
fn every_example_runs_and_its_checks_pass() {
    for e in EXAMPLES {
        let r = run(&e).unwrap_or_else(|why| panic!("{}: {why}", e.title));
        assert!(r.checks.iter().take(e.checks.len()).all(|c| c.starts_with("ok: ")), "{}: {:?}", e.title, r.checks);
        assert!(r.checks.last().unwrap().ends_with("all passed"), "{:?}", r.checks);
        assert!(!r.value.is_empty());
    }
}

#[test]
fn the_page_shows_the_command_line_programs_code() {
    for e in EXAMPLES {
        let (math, pipe, plain) = calls(&e);
        assert!(pipe.contains("flow:p_ipe<") && e.src.contains(&pipe), "{pipe}");
        assert!(e.src.contains(&plain) && !plain.contains("flow:"), "{plain}");
        assert_eq!(math.is_some(), e.math.is_some());
        let src = program(&e);
        assert!(src.starts_with(head(e.src)) && src.contains(&pipe));
        assert!(rebound(&src).is_empty(), "{}: {:?}", e.title, rebound(&src));
    }
    // A pipeline over lines is taken whole.
    assert!(statement(PIPELINES, "head").ends_with("|> it '+ '* i_nner V\"#"));
}

#[test]
fn each_expansion_is_the_pipelines_stages_as_code() {
    for e in EXAMPLES {
        let x = run(&e).unwrap().expansion;
        assert!(x.starts_with(&format!("{} := ", e.pipe)), "{x}");
        assert!(x.contains(":stage := ") && !x.contains("flow:p_ipe<"), "{x}");
        assert!(x.ends_with("} @)") && !x.contains("\n#"), "{x}");
    }
}

#[test]
fn the_cnn_layer_is_the_arithmetic_done_directly() {
    let e = EXAMPLES[0];
    let r = run(&e).unwrap();
    assert!(r.tex.as_deref().unwrap().starts_with("Y_{o,h,w} = "));
    // The picture: 0.1 everywhere, 0.9 on the 2 x 2 square at rows and
    // columns 2 and 3 (from 1); the filters and biases as in rosetta.xtl.
    let m = [0.0, 1.0, 1.0, 0.0, 0.0];
    let p = |i: usize, j: usize| 0.1 + 0.8 * m[i] * m[j];
    let f = [[1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0], [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0]];
    let b = [0.5, -0.25];
    assert!(ROSETTA.contains("1.0 0.0 -1.0 2.0 0.0 -2.0 1.0 0.0 -1.0 0.0 1.0 0.0 1.0 -4.0 1.0 0.0 1.0 0.0"));
    let mut want = Vec::new();
    for o in 0..2 {
        for h in 0..3 {
            for w in 0..3 {
                let s: f64 = (0..3).flat_map(|r| (0..3).map(move |c| (r, c))).map(|(r, c)| p(h + r, w + c) * f[o][3 * r + c]).sum();
                want.push((s + b[o]).max(0.0));
            }
        }
    }
    assert_eq!(r.value.len(), want.len());
    for (got, want) in r.value.iter().zip(&want) {
        assert!((got - want).abs() < 1e-9, "{:?} {:?}", r.value, want);
    }
}
