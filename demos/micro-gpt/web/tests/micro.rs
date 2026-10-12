use micro_gpt_web::micro::{chars, clean, head, program, run, uniforms, NAMES, SOURCE, TOKENS};

#[test]
fn it_invents_names_of_letters() {
    let r = run(0.5, "", &uniforms(1), 1).unwrap();
    assert_eq!(r.names.len(), NAMES);
    assert!(r.names.iter().all(|w| w.chars().all(|c| chars().contains(&c)) && w.len() <= 15), "{:?}", r.names);
    // The same numbers, the same names; others, others.
    assert_eq!(r.names, run(0.5, "", &uniforms(1), 1).unwrap().names);
    assert_ne!(r.names, run(0.5, "", &uniforms(2), 1).unwrap().names);
}

#[test]
fn names_begin_with_the_letters_given() {
    let r = run(0.5, "Ma!", &uniforms(3), 2).unwrap();
    assert!(r.names.iter().all(|w| w.starts_with("ma")), "{:?}", r.names);
    assert_eq!(clean("Ma!"), "ma");
}

#[test]
fn the_picked_names_probabilities_and_attention() {
    let r = run(0.5, "", &uniforms(4), 3).unwrap();
    let n = r.n;
    assert_eq!(r.probs.len(), n * TOKENS);
    for row in r.probs.chunks(TOKENS) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
    assert_eq!(r.heads.len(), 4);
    for h in &r.heads {
        // Causal: each row sums to 1, and nothing after its own place.
        for (i, row) in h.chunks(n).enumerate() {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(row[i + 1..].iter().all(|&w| w == 0.0));
        }
    }
}

#[test]
fn the_page_runs_the_programs_core() {
    assert!(SOURCE.contains(head()));
    let src = program(0.5, "ma", &uniforms(1), 1);
    assert!(microscope::source::rebound(&src).is_empty(), "{:?}", microscope::source::rebound(&src));
}
