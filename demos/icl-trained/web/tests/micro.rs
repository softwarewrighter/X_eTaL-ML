use icl_trained_web::micro::{class_of, gate, gate_tail, head, log, measured, prompt, run, tail, words, Edit, Set, MODELS, N, SOURCE};

/// The Rust model's logits at the query's arrow (data/<model>/expected.txt:
/// three unique-label prompts, then three recurring-label ones), its
/// weights rounded as written.
fn expected(model: &str, row: usize) -> Vec<f64> {
    let text = std::fs::read_to_string(format!("../data/{model}/expected.txt")).unwrap();
    text.lines().nth(row).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect()
}

#[test]
fn the_page_runs_the_command_line_programs_head() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:r_un := ") && head().contains("u:a_part := "));
    assert!(!head().contains("u:a_part \"mix\""), "the command line's own runs are left out");
}

#[test]
fn x_etal_gives_the_rust_models_logits() {
    for (m, (name, _, _)) in MODELS.iter().enumerate() {
        for (set, base) in [(Set::Unique, 0), (Set::Recurring, 3)] {
            for k in 1..=3 {
                let a = run(m, set, k, Edit::None, None).unwrap();
                let want = expected(name, base + k - 1);
                let worst = a.logits.iter().zip(&want).map(|(g, w)| (g - w).abs()).fold(0.0, f64::max);
                // The Rust model computes in f32 and its logits are written to 4 places.
                assert!(worst < 2e-3, "{name} {set:?} {k}: {worst}");
                assert!((a.margin - (a.logits[a.answer] - a.logits[a.alt])).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn a_slot_swapped_before_the_writers_is_answered() {
    // The mix model: swapped at the embedding, the swapped-in label wins;
    // after layer 3 the answer follows its key and stays.
    let intact = run(0, Set::Recurring, 1, Edit::None, None).unwrap();
    let early = run(0, Set::Recurring, 1, Edit::Swap(0), None).unwrap();
    let late = run(0, Set::Recurring, 1, Edit::Swap(3), None).unwrap();
    assert!(intact.margin > 2.0, "{}", intact.margin);
    assert!(early.margin < 0.0, "{}", early.margin);
    assert!(late.margin > 2.0, "{}", late.margin);
}

#[test]
fn the_gate_the_page_finds_is_the_command_lines() {
    // The gate model, as icl-trained.xtl printed it (reg/cli-icl-trained.out).
    let cli = &measured()[2];
    let g = gate(2).unwrap();
    for (i, (gap, acc)) in g.gaps.iter().enumerate() {
        assert!((gap - cli.gaps[i].0).abs() <= 0.005 + 1e-9, "gap after layer {}: {gap}", i + 1);
        assert!((acc - cli.gaps[i].1).abs() <= 0.005 + 1e-9, "accuracy after layer {}: {acc}", i + 1);
    }
    assert_eq!(g.layer(), cli.layer);
    assert!(g.directions.iter().all(|d| d.len() == 64 && (d.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() < 1e-9));
    // Pushing a recurring prompt toward unique along it lowers its margin
    // more than the same push in the random direction.
    let d = g.directions[g.layer() - 1].clone();
    let (mut along, mut random) = (0.0, 0.0);
    for k in 1..=N {
        along += run(2, Set::Recurring, k, Edit::Steer(g.layer(), -g.gaps[g.layer() - 1].0.abs()), Some(&d)).unwrap().margin;
        random += run(2, Set::Recurring, k, Edit::Random(g.layer(), -g.gaps[g.layer() - 1].0.abs()), Some(&d)).unwrap().margin;
    }
    assert!((along / N as f64 - cli.steer[0]).abs() <= 0.005 + 1e-9, "{}", along / N as f64);
    assert!((random / N as f64 - cli.steer[1]).abs() <= 0.005 + 1e-9, "{}", random / N as f64);
}

#[test]
fn the_command_lines_measures_and_the_logs_read() {
    let ms = measured();
    assert_eq!(ms.len(), 3);
    assert!(ms.iter().all(|m| m.swap.len() == 5 && m.readers.len() == 2 && m.gaps.len() == 3));
    for m in 0..3 {
        let l = log(m);
        assert!(l.len() > 20 && l[0].0 == 0.0);
        // Every run ends having learned recurring-label prompts.
        assert!(l.last().unwrap().2 > 4.0, "model {m}: {:?}", l.last());
    }
}

#[test]
fn every_sentence_has_its_one_class() {
    let cls = words();
    for set in [Set::Unique, Set::Recurring] {
        for k in 1..=N {
            let p = prompt(set, k);
            let mut i = 1;
            while i + 4 <= p.len() {
                assert!(class_of(&p[i..i + 3], &cls).is_some(), "{set:?} {k} at {i}");
                i += 6;
            }
        }
    }
}

#[test]
fn the_page_shows_every_line_it_runs() {
    for src in [format!("{}{}", head(), tail(2, Set::Recurring, 2, Edit::Swap(1))), format!("{}{}", head(), gate_tail(1)), format!("{}{}", head(), tail(0, Set::Unique, 3, Edit::Steer(3, -1.0)))] {
        assert!(microscope::source::rebound(&src).is_empty(), "names bound twice");
    }
}
