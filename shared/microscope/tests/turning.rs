//! The turning view's arithmetic (X_eTaL-manifolds' ask GM3): the
//! stated projection, the order things are drawn in, the drag, the keys
//! and the pitch's limit.

use microscope::turning::{key, marks, project, turn, typical, Angle, Mark, Point, Scene, Segment, PITCH};

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn the_projection_is_as_stated() {
    let front = Angle { yaw: 0.0, pitch: 0.0 };
    // The middle; x to the right and y up, in perspective 1 / 1.6.
    assert_eq!(project([0.0, 0.0, 0.0], front, 1.0, 520.0), (260.0, 260.0, 0.0));
    let (x, y, d) = project([1.0, 0.0, 0.0], front, 1.0, 520.0);
    assert!(near(x, 260.0 + 0.72 * 520.0 / 1.6) && near(y, 260.0) && near(d, 0.0));
    let (_, y, _) = project([0.0, 1.0, 0.0], front, 1.0, 520.0);
    assert!(near(y, 260.0 - 0.72 * 520.0 / 1.6), "y is up");
    // Toward the viewer (z) is nearer: a larger depth and a larger picture.
    let (x, _, d) = project([0.5, 0.0, 1.0], front, 1.0, 520.0);
    assert!(near(d, 1.0) && near(x, 260.0 + 0.72 * 520.0 * 0.5 / 1.25));
    // A quarter turn of yaw brings x to the viewer.
    let (x, _, d) = project([1.0, 0.0, 0.0], Angle { yaw: std::f64::consts::FRAC_PI_2, pitch: 0.0 }, 1.0, 520.0);
    assert!(near(x, 260.0) && near(d, -1.0), "x turns away: {x} {d}");
    // The scale divides; a point beyond 1.25 of it is pulled in.
    assert_eq!(project([2.0, 0.0, 0.0], front, 2.0, 520.0), project([1.0, 0.0, 0.0], front, 1.0, 520.0));
    assert_eq!(project([10.0, 0.0, 0.0], front, 1.0, 520.0), project([1.25, 0.0, 0.0], front, 1.0, 520.0));
}

#[test]
fn a_typical_radius_ignores_a_few_outliers() {
    let mut xyz: Vec<f64> = (0..20).flat_map(|_| [1.0, 0.0, 0.0]).collect();
    xyz.extend([100.0, 0.0, 0.0]);
    assert!(near(typical(&xyz), 1.15));
}

#[test]
fn wires_go_behind_and_the_nearest_on_top() {
    let scene = Scene {
        points: vec![
            Point { at: [0.0, 0.0, 1.0], color: "red".into(), label: "near".into() },
            Point { at: [0.0, 0.0, -1.0], color: "blue".into(), label: "far".into() },
        ],
        segments: vec![Segment { from: [0.0, 0.0, 0.0], to: [1.0, 0.0, 0.0], color: "black".into(), arrow: true }],
        wires: vec![vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
    };
    let m = marks(&scene, Angle { yaw: 0.0, pitch: 0.0 }, 1.0, 520.0);
    assert!(matches!(&m[0], Mark::Wire { path } if path.len() == 3));
    let order: Vec<String> = m[1..].iter().map(|k| match k {
        Mark::Point { i, .. } => format!("p{i}"),
        Mark::Segment { i, .. } => format!("s{i}"),
        Mark::Wire { .. } => "w".into(),
    }).collect();
    assert_eq!(order, ["p1", "s0", "p0"], "far point, segment at depth 0, near point");
    assert_eq!(scene.coordinates().len(), 3 * (2 + 2 + 3));
}

#[test]
fn a_drag_and_the_keys_turn_it_and_the_pitch_stops() {
    let a = Angle::default();
    assert_eq!(turn(a, 10.0, 0.0), Angle { yaw: a.yaw + 0.1, pitch: a.pitch });
    assert_eq!(turn(a, 0.0, 1000.0).pitch, PITCH);
    assert_eq!(turn(a, 0.0, -1000.0).pitch, -PITCH);
    assert_eq!(key(a, "ArrowRight"), Some(turn(a, 10.0, 0.0)));
    assert_eq!(key(a, "ArrowUp"), Some(turn(a, 0.0, -10.0)));
    assert_eq!(key(a, "x"), None);
}
