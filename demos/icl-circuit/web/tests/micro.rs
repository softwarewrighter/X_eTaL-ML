use icl_circuit_web::micro::{head, paper, presets, program, run, shown, Config, Prompt, KNOCKOUTS, SOURCE};

/// The authors' committed results (toy/toy_results.json in their
/// repository), to their printed digits.
const UNIQUE_SINK: f64 = 0.6682924705322446;
const UNIQUE_MARGIN: f64 = 0.24156734814027003;

fn preset(i: usize) -> Prompt {
    presets()[i].1.clone()
}

#[test]
fn the_page_runs_the_command_line_programs_core() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:r_un := ") && head().contains("u:r_ead := "));
}

#[test]
fn unique_labels_leave_the_readers_on_the_sink() {
    let a = run(&preset(0), &Config::default()).unwrap();
    assert!((a.sink - UNIQUE_SINK).abs() < 1e-12, "sink {}", a.sink);
    assert!((a.margin - UNIQUE_MARGIN).abs() < 1e-12, "margin {}", a.margin);
    assert!(a.gate < 1e-3, "the gate stays closed: {}", a.gate);
}

#[test]
fn a_repeated_label_opens_the_gate() {
    let a = run(&preset(1), &Config::default()).unwrap();
    // The authors average twenty shuffled orders; one order differs by
    // about 1e-6 (the duplicate detector leaks a little by position).
    assert!((a.sink - 0.020922554655484843).abs() < 1e-5, "sink {}", a.sink);
    assert!((a.margin - 0.8251299728418923).abs() < 1e-5, "margin {}", a.margin);
    assert!((a.gate - 0.5).abs() < 1e-3, "half the labels repeat: {}", a.gate);
    assert!(a.matching > 0.8);
}

#[test]
fn the_read_law_is_the_readers_softmax() {
    for (_, p) in presets() {
        for c in [Config::default(), Config { dup: false, ..Config::default() }, Config { gate_added: true, ..Config::default() }] {
            let a = run(&p, &c).unwrap();
            assert!((a.law - a.matching).abs() < 1e-14, "{}: {} against {}", p.text(), a.law, a.matching);
            // The same law from Rust: matches score 4, the sink 5 - 8 g, the rest 0.
            let (n, big_n) = (a.n() as f64, p.classes.len() as f64);
            let nm = p.classes.iter().filter(|&&c| c == p.query).count() as f64;
            let law = nm * 4f64.exp() / (nm * 4f64.exp() + (big_n - nm) + (n - big_n - 1.0) + (5.0 - 8.0 * a.gate).exp());
            assert!((law - a.matching).abs() < 1e-12);
        }
    }
}

#[test]
fn the_paper_knockouts() {
    let rows = paper().unwrap();
    assert_eq!(rows.len(), KNOCKOUTS.len());
    for ((name, pr, m, s), (mx, sx)) in KNOCKOUTS.iter().zip(&rows) {
        assert!((m - mx).abs() <= 0.005 + 1e-9, "{pr} {name}: margin {mx} against the paper's {m}");
        if let Some(s) = s {
            assert!((s - sx).abs() <= 0.005 + 1e-9, "{pr} {name}: sink {sx} against the paper's {s}");
        }
    }
    assert!(rows[1].1 < 0.001, "the gate added empties the sink");
}

#[test]
fn more_readers_each_capped() {
    let want = [(1, 0.16488017783668577), (3, 0.49118634252200316), (8, 1.2513313131215897), (20, 2.5262186825597706)];
    for (h, m) in want {
        let a = run(&preset(1), &Config { readers: h, capped: true, ..Config::default() }).unwrap();
        assert!((a.margin - m).abs() < 1e-5, "{h} readers: {}", a.margin);
    }
}

#[test]
fn the_blocks_hold_what_the_layers_wrote() {
    let a = run(&preset(1), &Config::default()).unwrap();
    for i in 0..a.n() {
        let (t, _, _) = a.tokens[i];
        assert_eq!(a.block(i, 2).iter().position(|&x| x == 1.0), Some(t), "type of token {i}");
        if t == 3 {
            let class = a.tokens[i - 2].1;
            assert!((a.block(i, 5)[class] - 1.0).abs() < 1e-9, "token {i}'s slot key is its sentence's class");
        }
    }
    // The second of each label is flagged a duplicate, the first not.
    let flags: Vec<f64> = (0..a.n()).filter(|&i| a.tokens[i].0 == 3).map(|i| a.block(i, 3)[0]).collect();
    assert!(flags[..3].iter().all(|&f| f < 1e-3) && flags[3..].iter().all(|&f| f > 0.999), "{flags:?}");
    let q = a.n() - 1;
    assert!((a.block(q, 6)[0] - 1.0).abs() < 1e-9, "the query code is the query's class");
}

#[test]
fn knocking_out_the_writers_leaves_nothing_to_read() {
    let a = run(&preset(1), &Config { writers: false, ..Config::default() }).unwrap();
    assert_eq!(a.margin, 0.0);
    // No slot has a key: the readers weigh every label alike, matching or not.
    let labels: Vec<f64> = (0..a.n()).filter(|&i| a.tokens[i].0 == 3).map(|i| a.attention[i]).collect();
    assert!(labels.iter().all(|&w| (w - labels[0]).abs() < 1e-15), "{labels:?}");
}

#[test]
fn prompts_read_and_refused() {
    let p = Prompt::parse("animal foo, color bar, Animal foo ? animal").unwrap();
    assert_eq!(p, Prompt::new(&[0, 1, 0], &[0, 1, 0], 0));
    assert_eq!(Prompt::parse(&p.text()).unwrap(), p);
    assert!(Prompt::parse("animal foo, color bar ? tool").unwrap_err().contains("nothing to retrieve"));
    assert!(Prompt::parse("animal zebra ? animal").unwrap_err().contains("not a label"));
    assert!(Prompt::parse("animal foo, color bar").is_err());
}

#[test]
fn the_page_shows_every_line_it_runs() {
    let p = preset(2);
    let c = Config::default();
    let src = program(&p, &c);
    assert!(microscope::source::rebound(&src).is_empty(), "names bound twice");
    let seen = shown(&p, &c);
    for line in src.lines() {
        assert!(seen.lines().any(|l| l == line), "not shown: {line}");
    }
    assert!(SOURCE.contains(icl_circuit_web::micro::paper_program().trim_start_matches(head())));
}
