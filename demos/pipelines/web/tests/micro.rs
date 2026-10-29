//! The page's model, natively: each example runs, agrees, and shows
//! code taken from the command-line programs; the CNN layer's maps are
//! checked against the arithmetic done directly in Rust.

use microscope::source::rebound;
use pipelines_web::micro::{calls, chosen_row, clickable, explore, first_pipeline, head, stage_ranges, STARTERS, program, run, statement, DIFFUSION, EQUATIONS, EXAMPLES, PIPELINES, ROSETTA};

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
        assert!(math.iter().all(|m| e.src.contains(m) && (m.contains("e:e_q<") || m.contains("flow:m_ath<"))), "{math:?}");
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
        // A line binding each stage's value (a label, or a fresh :stage name).
        let stages = calls(&e).1.matches("|>").count() + 1;
        let bound = x.lines().filter(|l| l.starts_with("  ") && !l.starts_with("   ") && l.contains(" := ")).count();
        assert!(bound >= stages && !x.contains("flow:p_ipe<"), "{}: {bound} of {stages}\n{x}", e.title);
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

#[test]
fn every_example_draws_a_pipeline_of_equations_that_computes() {
    for e in EXAMPLES {
        let (math, _, _) = calls(&e);
        let drawings: Vec<&String> = math.iter().filter(|m| m.contains("flow:m_ath<")).collect();
        assert_eq!(drawings.len(), 1, "{}: one pipeline drawn as math", e.title);
        // The drawing reads the very text of a pipeline the example computes and checks.
        let text = |s: &str, word: &str| s[s.find(word).unwrap() + word.len()..].to_string();
        let drawn = text(drawings[0], "flow:m_ath<");
        assert!(e.src.contains(&format!("flow:p_ipe<{drawn}")), "{}: no pipeline computes what is drawn", e.title);
        assert!(drawn.matches("e:e_q<").count() == drawn.matches("|>").count(), "{}: every stage an equation", e.title);
        let r = run(&e).unwrap();
        let tex = r.tex.iter().find(|t| t.starts_with("\\begin{aligned}")).unwrap();
        assert!(tex.ends_with("\\end{aligned}") && !tex.contains("\\texttt"), "{}: {tex}", e.title);
        assert_eq!(tex.matches("\\longrightarrow").count(), drawn.matches("|>").count(), "{}: {tex}", e.title);
    }
    let attention = run(&EXAMPLES.iter().find(|e| e.title == "Attention as equations").copied().unwrap()).unwrap();
    assert!(attention.tex[0].contains("Q_{i,d} = \\sum_{c=0}^{2} x_{i,c} \\, Wq_{c,d} \\qquad \\triangleright \\, q"), "{}", attention.tex[0]);
    let bert = run(&EXAMPLES.iter().find(|e| e.title == "A BERT layer").copied().unwrap()).unwrap();
    assert!(bert.tex[0].contains("G_{i,k} = \\frac{f_{i,k}}{1 + \\exp(-2 \\, u_{i,k})}") && bert.tex[0].contains("\\triangleright \\, out: n \\times 3"), "{}", bert.tex[0]);
    assert!(EQUATIONS.contains("Tiny BERT, demos/micro-bert"));
}

#[test]
fn every_stage_gives_its_value_and_the_last_is_the_pipelines() {
    use pipelines_web::micro::{clickable, expansion_ranges, stage_ranges};
    for e in EXAMPLES {
        let r = run(&e).unwrap_or_else(|why| panic!("{}: {why}", e.title));
        let (math, pipe, _) = calls(&e);
        let drawn = math.iter().find(|m| m.contains("flow:m_ath<")).unwrap();
        // A record per stage of the pipeline of equations, in order, the data first.
        let rows = drawn.matches("|>").count() + 1;
        assert_eq!(r.drawn.iter().map(|v| v.stage).collect::<Vec<_>>(), (1..=rows).collect::<Vec<_>>(), "{}", e.title);
        // Both end in the example's value; every record holds its items or its first 64.
        for trace in [&r.drawn, &r.stages] {
            let last = trace.last().unwrap();
            assert_eq!(last.items.len(), r.value.len().min(64), "{}", e.title);
            assert!(last.items.iter().zip(&r.value).all(|(a, b)| (a - b).abs() < 1e-9), "{}: {:?} {:?}", e.title, last.items, r.value);
            for v in trace.iter() {
                assert_eq!(v.items.len(), v.count().min(64), "{}: stage {}", e.title, v.stage);
                assert!(v.items.iter().all(|&x| v.min <= x && x <= v.max), "{}: stage {}", e.title, v.stage);
            }
        }
        // The pipeline's stages: a record each (a tuple's parts each), a range of its text each, a range of its expansion each.
        let stages = pipe.matches("|>").count() + 1;
        assert_eq!(r.stages.last().unwrap().stage, stages, "{}", e.title);
        let ranges = stage_ranges(&pipe);
        assert_eq!(ranges.len(), stages, "{}", e.title);
        assert_eq!(&pipe[ranges[1].0..ranges[1].0 + 1], pipe.split("|>").nth(1).unwrap().trim_start().get(..1).unwrap(), "{}", e.title);
        let lines = expansion_ranges(&r.expansion);
        assert_eq!(lines.len(), stages, "{}: {:?}", e.title, lines);
        assert!(r.expansion[lines[stages - 1].0..lines[stages - 1].1].contains(":="), "{}", e.title);
        // The math, clickable a row a stage, the chosen one marked.
        let c = clickable(&r.tex.iter().find(|t| t.starts_with("\\begin{aligned}")).unwrap(), Some(2));
        assert_eq!(c.matches("\\htmlClass{st st-").count(), rows, "{c}");
        assert_eq!(c.matches("st-on").count(), 1, "{c}");
        assert!(c.starts_with("\\begin{aligned} & \\htmlClass{st st-1}{"), "{c}");
    }
}

#[test]
fn a_click_chooses_the_row_under_it_whose_middle_is_nearest() {
    // Three rows; the third's fraction reaches up over the second's lower half.
    let rows = [(1, 0.0, 24.0), (2, 48.0, 72.0), (3, 60.0, 110.0)];
    assert_eq!(chosen_row(&rows, 10.0), Some(1));
    assert_eq!(chosen_row(&rows, 55.0), Some(2));
    assert_eq!(chosen_row(&rows, 70.0), Some(2));
    assert_eq!(chosen_row(&rows, 100.0), Some(3), "the last row is chosen too");
    assert_eq!(chosen_row(&rows, 30.0), None, "between rows, none");
    assert_eq!(chosen_row(&[], 5.0), None);
}

#[test]
fn compose_finds_the_first_pipeline_as_written() {
    let src = "x := 1 2 3\ny := @ flow:p_ipe< c\"x |> 2 * it $d |> d + \\\"a\\\" r_ho it\"\nz := 1";
    let (call, lit) = first_pipeline(src).unwrap();
    assert_eq!(call, "flow:p_ipe< c\"x |> 2 * it $d |> d + \\\"a\\\" r_ho it\"");
    assert_eq!(lit, "c\"x |> 2 * it $d |> d + \\\"a\\\" r_ho it\"");
    // The verbatim form; a pipeline before a drawing preferred.
    let src = "d := @ flow:m_ath< c#\"X |> \"Y[i] | 2\" e:e_q< \"it[i]\"\"#\np := @ flow:p_ipe< c#\"X |> 'n_eg\"#";
    assert_eq!(first_pipeline(src).unwrap().1, "c#\"X |> 'n_eg\"#");
    assert_eq!(first_pipeline("d := @ flow:m_ath< c#\"X |> 'n_eg\"#").unwrap().0, "flow:m_ath< c#\"X |> 'n_eg\"#");
    assert_eq!(first_pipeline("1 + 2"), None);
    assert_eq!(first_pipeline("@ flow:p_ipe< c\"x |> 2 * it"), None, "an unclosed string");
}

#[test]
fn every_starter_with_a_pipeline_shows_its_stages() {
    let mut shown = 0;
    for (title, src) in STARTERS {
        let Some(found) = explore(src) else { continue };
        let x = found.unwrap_or_else(|e| panic!("{title}: {e}"));
        let n = stage_ranges(&x.call).len();
        assert!(n >= 2, "{title}: {n} stages");
        assert_eq!(x.trace.iter().map(|v| v.stage).max(), Some(n), "{title}: a value for every stage");
        assert_eq!(clickable(&x.tex, None).matches("\\htmlClass{st st-").count(), n, "{title}: a drawn row a stage");
        shown += 1;
    }
    assert_eq!(shown, 2, "the pipeline and the pipeline of equations");
    // The pipeline of equations: Y = max(X W, 0).
    let x = explore(STARTERS[2].1).unwrap().unwrap();
    let y = x.trace.iter().find(|v| v.stage == 3).unwrap();
    assert_eq!((y.shape.clone(), y.items.clone()), (vec![3, 2], vec![3.0, 0.0, 0.0, 10.0, 0.0, 0.0]));
}

#[test]
fn compose_says_why_a_pipeline_inside_a_function_cannot_be_shown() {
    let src = "\"flow:\" u_se< \"Flow\"\nu:f := { v -> @ flow:p_ipe< c\"v |> 2 * it\" }\nu:f 1 2 3\n";
    assert!(matches!(explore(src), Some(Err(_))));
    assert_eq!(explore("1 + 2"), None);
}

#[test]
fn compose_reads_an_integral_inside_an_equation() {
    // The user's program, 2026-10-29: GELU as x times the Gaussian CDF,
    // the integral read by Equations (X_eTaL-libraries past 3a61a22).
    let src = "\"e:\" u_se< \"Equations\"\npi := 3.141592653589793\nu:g_elu := { x -> x * 0.5 + \"L[] | \" e:e_q< \"integral u:=0..x by simpson 32 | exp(-(u^2) / 2) / sqrt(2 * pi)\" }\n'u:g_elu e_ach 1.0 2.0\n";
    let c = microscope::palette::compose(src);
    assert_eq!(c.err, "");
    let g: Vec<f64> = c.out[0].split_whitespace().map(|w| w.parse().unwrap()).collect();
    assert!((g[0] - 0.8413447).abs() < 1e-6 && (g[1] - 1.9544997).abs() < 1e-6, "{g:?}");
}
