use micro_bert_web::micro::{accuracy, chars, clean, head, program, run, LETTERS, SOURCE};

#[test]
fn a_name_gets_one_blank() {
    assert_eq!(clean("Ma_y!"), "ma_y");
    assert_eq!(clean("mary"), "mary_");
    assert_eq!(clean("m_a_y"), "m_ay");
    assert_eq!(clean(""), "_");
    assert_eq!(clean("abcdefghijklmnopqrst").chars().count(), 16);
    assert!(clean("abcdefghijklmnopqrst").ends_with('_'));
}

#[test]
fn the_blank_gets_a_probability_per_letter() {
    let f = run("ma_y").unwrap();
    assert_eq!((f.name.as_str(), f.n), ("ma_y", 4));
    assert_eq!(f.probs.len(), LETTERS);
    assert!((f.probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert_eq!(chars().len(), LETTERS);
}

#[test]
fn the_page_fills_the_blank_as_the_rust_model_scores_it() {
    // The trainer's scores for "em_a" (data/expected.txt), row 3 the blank, through softmax.
    let expected: Vec<f64> = include_str!("../../data/expected.txt").lines().nth(2).unwrap().split_whitespace().map(|w| w.parse().unwrap()).collect();
    let top = expected.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = expected.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = e.iter().sum();
    let f = run("em_a").unwrap();
    for (p, q) in f.probs.iter().zip(e.iter().map(|x| x / total)) {
        assert!((p - q).abs() < 1e-9, "{p} {q}");
    }
}

#[test]
fn every_head_looks_both_ways() {
    let f = run("sophi_").unwrap();
    assert_eq!(f.heads.len(), 4);
    for h in &f.heads {
        assert_eq!(h.len(), 36);
        for (i, row) in h.chunks(6).enumerate() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            // Not causal: a place gives weight to places after it.
            if i < 5 {
                assert!(row[i + 1..].iter().any(|&w| w > 0.0));
            }
        }
    }
}

#[test]
fn the_model_is_drawn_as_math() {
    let f = run("_").unwrap();
    assert_eq!(f.tex.len(), 4);
    assert!(f.tex[0].starts_with("\\mathrm{BERT}(x) = "), "{}", f.tex[0]);
    assert!(f.tex[1].starts_with("S_{i,j} = \\frac{\\sum_{k=0}^{7}"), "{}", f.tex[1]);
    assert!(f.tex[3].starts_with("Y_{i,d} = \\sum_{j=0}^{n - 1} A_{i,j}"), "{}", f.tex[3]);
}

#[test]
fn it_does_better_than_guessing_on_names_it_never_saw() {
    let [right, three, total, guess] = accuracy();
    assert!(total > 5000 && right > guess && three > right, "{right} {three} {total} {guess}");
}

#[test]
fn the_page_runs_the_programs_core() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:l_ayer := { (H, l) ->") && head().contains("@ flow:p_ipe<"));
    let src = program("ma_y");
    assert!(microscope::source::rebound(&src).is_empty(), "{:?}", microscope::source::rebound(&src));
}
