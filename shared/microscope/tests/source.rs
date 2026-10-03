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
