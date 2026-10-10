use k_means_web::micro::{head, k_step, program, run, Start, POINTS, SOURCE};

/// k-means in Rust, plainly: the nearest center, then the means.
fn direct(points: &[f64], centers: &[f64]) -> Vec<f64> {
    let k = centers.len() / 2;
    let (mut sum, mut count) = (vec![0.0; 2 * k], vec![0.0; k]);
    for p in points.chunks(2) {
        let d = |j: usize| (p[0] - centers[2 * j]).powi(2) + (p[1] - centers[2 * j + 1]).powi(2);
        let j = (0..k).fold(0, |b, j| if d(j) < d(b) { j } else { b });
        sum[2 * j] += p[0];
        sum[2 * j + 1] += p[1];
        count[j] += 1.0;
    }
    (0..2 * k).map(|i| if count[i / 2] > 0.0 { sum[i] / count[i / 2] } else { centers[i] }).collect()
}

#[test]
fn a_step_is_k_means_as_rust_writes_it() {
    let a = run(5, Start::Poor, None, &[]).unwrap();
    assert_eq!(a.points.len(), 2 * POINTS);
    let mut c = a.centers.clone();
    let mut history = vec![a.inertia];
    for _ in 0..4 {
        let b = run(5, Start::Poor, Some(&c), &history).unwrap();
        let want = direct(&a.points, &c);
        assert!(b.centers.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-12), "{:?} {:?}", b.centers, want);
        assert!(b.inertia <= *history.last().unwrap() + 1e-9, "the inertia never rises");
        history.push(b.inertia);
        c = b.centers;
    }
}

#[test]
fn the_poor_start_settles_worse_than_farthest_first() {
    let settle = |start| {
        let mut a = run(5, start, None, &[]).unwrap();
        for _ in 0..20 {
            let c = a.centers.clone();
            a = run(5, start, Some(&c), &[]).unwrap();
            if a.settled {
                break;
            }
        }
        a
    };
    let (poor, far) = (settle(Start::Poor), settle(Start::Farthest));
    assert!(poor.settled && far.settled);
    assert!(poor.inertia > 10.0 * far.inertia, "{} vs {}", poor.inertia, far.inertia);
    let mut used = far.points_in.clone();
    used.sort();
    used.dedup();
    assert_eq!(used, [1, 2, 3, 4, 5], "farthest-first: every center has points");
}

#[test]
fn the_page_runs_the_programs_core_and_quotes_learns_step() {
    assert!(SOURCE.contains(head()));
    assert!(k_step().starts_with("l:k_step := { X C ->") && k_step().ends_with('}'));
    for src in [program(5, Start::Poor, None, &[]), program(3, Start::Farthest, Some(&[0.0; 6]), &[1.0])] {
        assert!(microscope::source::rebound(&src).is_empty(), "{:?}", microscope::source::rebound(&src));
    }
}
