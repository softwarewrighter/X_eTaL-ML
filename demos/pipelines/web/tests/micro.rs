//! The page's model, natively: each example runs, agrees, and shows
//! code taken from the command-line programs; the CNN layer's maps are
//! checked against the arithmetic done directly in Rust.

use microscope::source::rebound;
use pipelines_web::micro::{calls, head, program, run, statement, DIFFUSION, EXAMPLES, PIPELINES, ROSETTA};

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
        assert_eq!(math.len(), e.math.len());
        assert!(math.iter().all(|m| e.src.contains(m) && m.contains("e:e_q<")), "{math:?}");
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
    assert!(r.tex[0].starts_with("Y_{o,h,w} = "), "{:?}", r.tex);
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

#[test]
fn the_diffusion_sampler_is_the_arithmetic_done_directly() {
    let e = EXAMPLES[1];
    assert_eq!(e.src, DIFFUSION);
    let r = run(&e).unwrap();
    assert!(r.tex[0].starts_with("Zn_{n,k} = a_{t-1}") && r.tex[1].starts_with("z0_{n,k} = zT_{n,k} + "), "{:?}", r.tex);
    // The data as diffusion.xtl makes it (r_ange counts from 1): the
    // noise of every step, the latents of noise, the networks' weights.
    let (n, k, t) = (2usize, 4usize, 3usize);
    let a = [0.9, 0.8, 0.7];
    let b = [0.1, 0.2, 0.3];
    let sigma = [0.05, 0.05, 0.0];
    let noise = |s: usize, i: usize, j: usize| 0.5 * (((s * n + i) * k + j + 1) as f64).cos();
    let we = |i: usize, j: usize| 0.3 * ((i * 4 + j + 1) as f64).sin();
    let wd = |i: usize, j: usize| 0.3 * ((i * 4 + j + 1) as f64).cos();
    let mut z: Vec<Vec<f64>> = (0..n).map(|i| (0..k).map(|j| ((i * k + j + 1) as f64).sin()).collect()).collect();
    for level in (1..=t).rev() {
        let mut next = z.clone();
        for i in 0..n {
            // The denoiser: the latent, its level and the prompt's embedding through a dense layer.
            let joined: Vec<f64> = z[i].iter().copied().chain([level as f64, 0.5, -0.5]).collect();
            for j in 0..k {
                let eps = (0..7).map(|q| joined[q] * we(q, j)).sum::<f64>() + we(7, j);
                next[i][j] = a[level - 1] * z[i][j] + b[level - 1] * eps + sigma[level - 1] * noise(level - 1, i, j);
            }
        }
        z = next;
    }
    let mut want = Vec::new();
    for i in 0..n {
        for j in 0..k {
            want.push((0..k).map(|q| z[i][q] * wd(q, j)).sum::<f64>() + wd(4, j));
        }
    }
    assert_eq!(r.value.len(), want.len());
    for (got, want) in r.value.iter().zip(&want) {
        assert!((got - want).abs() < 1e-9, "{:?} {:?}", r.value, want);
    }
}

#[test]
fn what_compose_starts_from_runs_and_the_equation_one_draws_its_math() {
    use microscope::palette::compose;
    use pipelines_web::micro::STARTERS;
    for (title, src) in STARTERS {
        let c = compose(src);
        assert_eq!(c.err, "", "{title}: {}", c.err);
        assert!(!c.out.is_empty(), "{title}: printed nothing");
        assert!(c.expansion.unwrap().is_ok(), "{title}");
        assert!(rebound(src).is_empty(), "{title}");
    }
    let c = compose(STARTERS[0].1);
    assert_eq!(c.tex.len(), 1);
    assert!(c.tex[0].starts_with("z_{i} = "), "{:?}", c.tex);
    assert_eq!(c.out, ["16.0 35.0"]);
}
