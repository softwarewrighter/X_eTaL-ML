use microscope::source::{between, find, NONE};

#[test]
fn ranges() {
    let src = "a := 1\nb := { x ->\n  x\n}\nc := 2\n";
    assert_eq!(&src[between(src, "b := ", "\n}").0..between(src, "b := ", "\n}").1], "b := { x ->\n  x\n}");
    assert_eq!(&src[find(src, "c := 2").0..find(src, "c := 2").1], "c := 2");
    assert_eq!(find(src, "zzz"), NONE);
}

#[test]
fn a_range_to_a_marker_on_its_own_line_ends_after_it() {
    let src = "x := 1\ny := 2\nz := 3\n";
    let (a, b) = between(src, "x := ", "y := ");
    assert_eq!(&src[a..b], "x := 1\ny := 2");
    let (a, b) = between(src, "z := ", "z := ");
    assert_eq!(&src[a..b], "z := 3");
}

/// One lambda closed and another opened on the same line are two
/// scopes (scripts/check-rebind.py, the same rule).
#[test]
fn two_lambdas_on_one_line_are_two_scopes() {
    let src = "c := ({ @ ->\n  x := 1\n  x\n} @) + ({ @ ->\n  x := 2\n  x\n} @)\n";
    assert!(microscope::source::rebound(src).is_empty(), "{:?}", microscope::source::rebound(src));
    assert_eq!(microscope::source::rebound("u:f_ := { x ->\n  x := x + 1\n  x\n}\n").len(), 1);
}
