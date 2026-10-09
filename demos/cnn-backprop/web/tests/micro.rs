use cnn_backprop_web::micro::{head, program, run, test_digits, PARTS, SHOWN, SOURCE};

#[test]
fn a_run_reads_back_the_state_it_wrote() {
    let a = run(None, 0, &[]).unwrap();
    assert_eq!(a.state.iter().map(|p| p.len()).collect::<Vec<_>>(), PARTS.iter().map(|p| p.1 * p.2).collect::<Vec<_>>());
    assert!(a.right < 0.3, "untrained: {}", a.right);
    // Two steps at once equal one and one, carried through the page's files.
    let two = run(Some(&a.state), 2, &[]).unwrap();
    let one = run(Some(&a.state), 1, &[]).unwrap();
    let more = run(Some(&one.state), 1, &[(a.loss, a.right)]).unwrap();
    assert_eq!(two.state, more.state, "the state goes through the page's files exactly");
    assert_eq!(more.state[9], vec![2.0], "two steps counted");
    assert_eq!(more.reads.len(), SHOWN);
    assert!(more.pictures.len() == 1 && more.pictures[0].contains("test share right"));
}

#[test]
fn it_learns_the_digits() {
    let mut a = run(None, 0, &[]).unwrap();
    let start = a.right;
    for _ in 0..10 {
        a = run(Some(&a.state), 2, &[]).unwrap();
    }
    assert!(a.right > 0.75 && a.right > start + 0.5, "after 20 steps: {} right (from {start})", a.right);
    let digits = test_digits();
    let right = digits.iter().zip(&a.reads).filter(|((l, _), r)| l == *r).count();
    assert!(right >= 14, "{right} of {SHOWN} read right");
}

#[test]
fn the_page_runs_the_command_line_programs_core() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:g_rad := { (K, B, W) i ->") && head().contains("u:s_tep := {"));
    assert!(program(true, 2, &[]).contains("s := (8 9 r_eshape n_umbers []N_GET \"state/K.txt\", n_umbers []N_GET \"state/B.txt\""));
}
