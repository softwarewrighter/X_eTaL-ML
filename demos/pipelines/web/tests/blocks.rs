//! A stage in 3D: its inputs by Flow's rule, a value as a block of cubes.

use pipelines_web::blocks::{block, dragged, inputs, picture, CAP, SIDE, SIDE_RANGE, SIZE};
use pipelines_web::micro::{calls, run, stage_ranges, EXAMPLES};

fn labels(pairs: &[(usize, &str)]) -> Vec<(usize, String)> {
    pairs.iter().map(|&(s, l)| (s, l.to_string())).collect()
}

#[test]
fn a_stage_reads_it_its_labels_or_as_a_function_the_previous_value() {
    let ls = labels(&[(1, "x"), (2, "q"), (3, "k"), (4, "v"), (5, "s"), (6, "a")]);
    // A function applied to the previous value; an expression of it.
    assert_eq!(inputs("cv:w_indows", 3, &ls), vec![2]);
    assert_eq!(inputs("(1, B) nn:a_ddAxis it", 4, &ls), vec![3]);
    // Equations: the arrays named in the strings, followed by [; the
    // target's string and the index names (k is a label too) left out.
    assert_eq!(inputs(r#""K[j,d] | n 2" e:e_q< "sum c:=0..2 | x[j,c] * Wk[c,d]" $k"#, 3, &ls), vec![1]);
    assert_eq!(inputs(r#""S[i,j] | n n" e:e_q< "(sum d:=0..1 | q[i,d] * k[j,d]) / sqrt(2.0)" $s [n n]"#, 5, &ls), vec![2, 3]);
    assert_eq!(inputs(r#""Y[i,d] | n 2" e:e_q< "sum j:=0..n-1 | a[i,j] * v[j,d]""#, 7, &ls), vec![4, 6]);
    assert_eq!(inputs(r#""F[i,k] | n 6" e:e_q< "(sum c:=0..2 | it[i,c] * W1[c,k]) + W1[3,k]""#, 7, &ls), vec![6]);
    // Labels and it together; a later label is not a read; the data none.
    assert_eq!(inputs("e / '+ r_/ e", 3, &labels(&[(2, "e")])), vec![2]);
    assert_eq!(inputs("it + s", 4, &labels(&[(1, "s")])), vec![1, 3]);
    assert_eq!(inputs("act:r_elu Y", 6, &labels(&[(5, "Y")])), vec![5]);
    assert_eq!(inputs("X $x", 1, &ls), Vec::<usize>::new());
}

#[test]
fn every_stage_of_every_example_reads_only_earlier_stages() {
    for e in EXAMPLES {
        let r = run(&e).unwrap();
        let (math, pipe, _) = calls(&e);
        let drawn = math.iter().find(|c| c.contains("flow:m_ath<")).unwrap();
        for (text, trace) in [(drawn.as_str(), &r.drawn), (pipe.as_str(), &r.stages)] {
            let ls: Vec<(usize, String)> = trace.iter().filter(|v| !v.label.is_empty()).map(|v| (v.stage, v.label.clone())).collect();
            for (k, &(a, b)) in stage_ranges(text).iter().enumerate() {
                let ins = inputs(&text[a..b], k + 1, &ls);
                assert!(ins.iter().all(|&i| i >= 1 && i <= k), "{}: stage {} reads {ins:?}", e.title, k + 1);
                assert_eq!(ins.is_empty(), k == 0, "{}: stage {} reads {ins:?}", e.title, k + 1);
            }
        }
    }
}

#[test]
fn a_shape_is_a_block_of_slices_rows_and_columns() {
    assert_eq!(block(&[]), (1, 1, 1));
    assert_eq!(block(&[5]), (1, 1, 5));
    assert_eq!(block(&[3, 4]), (1, 3, 4));
    assert_eq!(block(&[2, 3, 4]), (2, 3, 4));
    assert_eq!(block(&[2, 5, 3, 4]), (10, 3, 4));
}

#[test]
fn a_block_shows_its_known_items_inside_the_picture_the_nearest_last() {
    let p = picture(&[2, 3, 4], 24, 0.6, 0.45);
    // Every item, three faces each from this side; 12 outline edges.
    let mut items: Vec<usize> = p.faces.iter().map(|f| f.item).collect();
    items.dedup();
    assert_eq!(p.faces.len(), 24 * 3);
    assert_eq!({ let mut s = items.clone(); s.sort(); s.dedup(); s.len() }, 24);
    assert_eq!(p.edges.len(), 12);
    assert!(p.faces.windows(2).all(|w| w[0].depth <= w[1].depth), "the farthest first");
    for f in &p.faces {
        for &(x, y) in &f.pts {
            assert!((0.0..=SIZE).contains(&x) && (0.0..=SIZE).contains(&y), "{x} {y} outside");
        }
    }
    // The labels: columns, rows, slices.
    let text: Vec<&str> = p.labels.iter().map(|l| l.2.as_str()).collect();
    assert_eq!(text, ["4", "3", "2"]);
}

#[test]
fn a_large_value_is_drawn_capped_its_unknown_items_left_out() {
    // 64 items known of 20 x 30: the first two rows and 4 of the third.
    let p = picture(&[20, 30], 64, 0.6, 0.45);
    let mut items: Vec<usize> = p.faces.iter().map(|f| f.item).collect();
    items.sort();
    items.dedup();
    assert_eq!(items, (0..CAP).chain(30..30 + CAP).chain(60..64).collect::<Vec<_>>());
    let text: Vec<&str> = p.labels.iter().map(|l| l.2.as_str()).collect();
    assert_eq!(text, ["30 (12 drawn)", "20 (12 drawn)"]);
    // A number: one cube, no labels.
    let one = picture(&[], 1, 0.6, 0.45);
    assert_eq!(one.faces.len(), 3);
    assert!(one.labels.is_empty());
}

#[test]
fn turning_shows_other_faces() {
    let front = picture(&[1, 1, 1], 1, 0.0, 0.0);
    assert_eq!(front.faces.len(), 1, "head on, the front face only");
    let above = picture(&[1, 1, 1], 1, 0.0, 0.5);
    assert_eq!(above.faces.len(), 2, "from above, the top too");
}

#[test]
fn dragging_the_divider_left_widens_the_pane_within_its_range() {
    assert_eq!(dragged(SIDE, 1000.0, 800.0), SIDE + 200.0);
    assert_eq!(dragged(SIDE, 1000.0, 1100.0), SIDE - 100.0);
    assert_eq!(dragged(SIDE, 1000.0, 1900.0), SIDE_RANGE.0, "never narrower than its least");
    assert_eq!(dragged(SIDE, 1000.0, 0.0), SIDE_RANGE.1, "never wider than its most");
}

#[test]
fn the_chain_rules_backward_stages_read_the_forward_values_they_need() {
    // 1 X, 2 Z, 3 Y, 4 L, 5 dL/dy, 6 dL/dz, 7 dL/dW.
    let e = EXAMPLES.iter().find(|e| e.title == "The chain rule").unwrap();
    let r = run(e).unwrap();
    let (_, pipe, _) = calls(e);
    let ls: Vec<(usize, String)> = r.stages.iter().filter(|v| !v.label.is_empty()).map(|v| (v.stage, v.label.clone())).collect();
    let rs = stage_ranges(&pipe);
    let reads = |k: usize| inputs(&pipe[rs[k - 1].0..rs[k - 1].1], k, &ls);
    assert_eq!(rs.len(), 7);
    assert_eq!(reads(5), vec![3], "dL/dy reads y (T is data)");
    assert_eq!(reads(6), vec![3, 5], "dL/dz reads y for its derivative and the gradient before it");
    assert_eq!(reads(7), vec![1, 6], "dL/dW reads x and the gradient before it");
    let last = r.stages.iter().find(|v| v.stage == 7).unwrap();
    assert_eq!(last.shape, vec![3, 2]);
}
