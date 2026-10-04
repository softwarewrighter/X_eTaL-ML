use attention_web::micro::{head, program, run, tokens, DATA, SOURCE};

fn data(path: &str) -> Vec<f64> {
    DATA.iter().find(|d| d.0 == path).unwrap().1.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

const TIRED: &str = "the animal did not cross the street because it was tired";

#[test]
fn the_page_runs_the_command_line_programs_head() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:a_ttend := ") && head().contains("[]N_GET \"data/wq.txt\""));
}

#[test]
fn the_scores_and_weights_are_the_algebra() {
    let a = run(TIRED, false).unwrap();
    let (f, wq, wk) = (data("data/features.txt"), data("data/wq.txt"), data("data/wk.txt"));
    let ids: Vec<usize> = tokens(TIRED).iter().map(|t| t.1 - 1).collect();
    let n = ids.len();
    let proj = |w: &[f64], i: usize, d: usize| (0..8).map(|k| f[ids[i] * 8 + k] * w[k * 2 + d]).sum::<f64>();
    for i in 0..n {
        let mut row = vec![0.0; n];
        for j in 0..n {
            let s = (proj(&wq, i, 0) * proj(&wk, j, 0) + proj(&wq, i, 1) * proj(&wk, j, 1)) / 2f64.sqrt();
            assert!((a.s[i * n + j] - s).abs() < 1e-9, "score {i} {j}");
            row[j] = s;
        }
        let mx = row.iter().cloned().fold(f64::MIN, f64::max);
        let sum: f64 = row.iter().map(|v| (v - mx).exp()).sum();
        for j in 0..n {
            assert!((a.a[i * n + j] - (row[j] - mx).exp() / sum).abs() < 1e-9, "weight {i} {j}");
        }
        let y: Vec<f64> = (0..8).map(|k| (0..n).map(|j| a.a[i * n + j] * f[ids[j] * 8 + k]).sum()).collect();
        assert!(y.iter().zip(&a.y[i * 8..i * 8 + 8]).all(|(p, q)| (p - q).abs() < 1e-9), "output {i}");
    }
}

#[test]
fn adjectives_find_what_they_describe() {
    let a = run(TIRED, false).unwrap();
    assert_eq!(a.top(10).map(|j| a.words[j].as_str()), Some("animal"), "tired");
    assert_eq!(a.top(8).map(|j| a.words[j].as_str()), Some("animal"), "it");
    assert_eq!(a.top(0), None, "the looks for nothing");
    let b = run("the animal did not cross the street because it was wide", false).unwrap();
    assert_eq!(b.top(10).map(|j| b.words[j].as_str()), Some("street"), "wide");
}

#[test]
fn the_causal_mask_hides_later_words() {
    let a = run(TIRED, true).unwrap();
    let n = a.n();
    for i in 0..n {
        assert!((i + 1..n).all(|j| a.a[i * n + j] < 1e-12), "row {i}");
        assert!((a.a[i * n..(i + 1) * n].iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }
}

#[test]
fn unknown_words_are_question_marks() {
    let t = tokens("The zebra crossed the roads.");
    assert_eq!(t.iter().map(|x| x.1).collect::<Vec<_>>(), vec![1, 21, 21, 1, 9]);
}

#[test]
fn the_page_shows_every_line_it_runs() {
    use attention_web::view::program as shown;
    let ids = [1, 3, 12, 15, 13, 1, 7, 20, 11, 14, 16];
    for causal in [false, true] {
        let s = shown(&ids, causal);
        for line in program(&ids, causal).lines() {
            assert!(s.lines().any(|l| l == line), "not shown: {line}");
        }
    }
}
