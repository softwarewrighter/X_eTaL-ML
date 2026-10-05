use xetal_probe::run;

#[test]
fn evaluates_a_reduction() {
    assert_eq!(run("'+ r_/_2 2 3 r_eshape r_ange 6"), ("6 15\n".to_string(), String::new()));
}

#[test]
fn steps_the_life_line() {
    let src = "u:l_ife := { ('+ r_/_12 -1 0 1 o_-_12 _r) { (_l = 3) + _r * _l = 4 } _r }\n\
               u:l_ife 5 5 r_eshape 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0";
    let (out, err) = run(src);
    assert_eq!(err, "");
    assert_eq!(out, "0 0 0 0 0\n0 0 0 0 0\n0 1 1 1 0\n0 0 0 0 0\n0 0 0 0 0\n");
}

#[test]
fn reports_an_error_instead_of_panicking() {
    let (_, err) = run("1 +");
    assert!(!err.is_empty());
}
