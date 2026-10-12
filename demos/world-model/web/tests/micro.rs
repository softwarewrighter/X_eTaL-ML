use world_model_web::micro::{begin, exact, head, held_out, step, Place, State, DATA, SIDE, SOURCE, TOP};
use world_model_web::view::program;

/// The world in Rust, plainly: move, and bounce off a wall.
fn bounce(p: i64, v: i64) -> (i64, i64) {
    match p + v {
        n if n < 0 => (-n, -v),
        n if n > TOP => (2 * TOP - n, -v),
        n => (n, v),
    }
}

fn advance(s: State) -> State {
    let ((x, vx), (y, vy)) = (bounce(s[0], s[2]), bounce(s[1], s[3]));
    [x, y, vx, vy]
}

#[test]
fn the_first_run_is_the_world_as_rust_writes_it() {
    for s in [[0, 0, -2, -1], [10, 9, 2, 2], [1, 5, -1, 1], [5, 10, 1, 2], [3, 7, 2, -1]] {
        let b = begin(s).unwrap();
        let (s1, s2) = (advance(s), advance(advance(s)));
        let want: [Place; 3] = [(s[0], s[1]), (s1[0], s1[1]), (s2[0], s2[1])];
        assert_eq!(b.window, want, "{s:?}");
        assert_eq!(b.truth, advance(s2), "{s:?}");
    }
}

#[test]
fn a_prediction_is_the_trainers_rust_model() {
    // The pixels data/expected.txt holds for the window of (3, 7, 2, -1).
    let b = begin([3, 7, 2, -1]).unwrap();
    let st = step(&b.window, b.truth).unwrap();
    let want: Vec<f64> = include_str!("../../data/expected.txt").split_whitespace().map(|t| t.parse().unwrap()).collect();
    assert_eq!(want.len(), SIDE * SIDE);
    let off = st.probs.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(off < 1e-6, "{off}");
    assert_eq!(st.next, advance(b.truth));
}

#[test]
fn the_dream_from_the_first_unseen_state_tracks_the_ball_for_12_steps() {
    // As the command line's dream does (its last line: 12).
    let s = held_out()[0];
    let b = begin(s).unwrap();
    let (mut window, mut truth) = (b.window, b.truth);
    let mut exact_frames = 0;
    for _ in 0..12 {
        let st = step(&window, truth).unwrap();
        assert_eq!(st.place, (truth[0], truth[1]), "the dreamed ball is where the true one is");
        exact_frames += exact(&st.probs, truth) as usize;
        window = [window[1], window[2], st.place];
        truth = st.next;
    }
    assert!(exact_frames >= 8, "{exact_frames} of 12 frames exact");
    assert_eq!(held_out().len(), 387);
    assert_eq!(DATA.len(), 5);
}

#[test]
fn the_page_runs_the_programs_core() {
    assert!(SOURCE.contains(head()));
    assert!(head().contains("u:p_lace := {"), "the likeliest place is in the core, for the page");
    for src in [program([3, 7, 2, -1], &[(0, 0); 3], [0, 0, 1, 1], false), program([3, 7, 2, -1], &[(3, 7), (5, 6), (7, 5)], [9, 4, 2, -1], true)] {
        assert!(microscope::source::rebound(&src).is_empty(), "{:?}", microscope::source::rebound(&src));
    }
}
