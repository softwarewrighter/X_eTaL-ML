use microscope::run::{lit, lit_or_zero, matrix, numbers, output, section};

#[test]
fn literals_have_no_exponent_and_read_back() {
    for x in [1.0, -0.6, 3.0 / 65536.0, -1.234e-11, 2.5e-300, 123456.789, 0.1 + 0.2] {
        let s = lit(x);
        assert!(!s.contains('e') && s.contains('.'), "{x} as {s}");
        assert_eq!(s.parse::<f64>().unwrap(), x, "{x} as {s}");
    }
    assert_eq!(lit(0.0), "0.0");
    assert_eq!(lit(-0.8), "-0.8");
    assert_eq!(lit(3.0), "3.0");
    assert_eq!(lit_or_zero(3e-20, 1e-15), "0.0");
}

#[test]
fn literals_run_in_x_etal() {
    let x = 4.57763671875e-5;
    let out = output(&format!("{} * 2.0", lit(x)), 1).unwrap();
    assert_eq!(numbers::<f64>(&out[0], 1).unwrap(), vec![x * 2.0]);
}

#[test]
fn a_matrix_goes_in_and_comes_back() {
    let m = matrix("m", 2, 3, (1..=6).map(|i| i.to_string()));
    let out = output(&format!("{m}r_avel m\ns_hape m\n"), 2).unwrap();
    assert_eq!(numbers::<i64>(&out[0], 6).unwrap(), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(numbers::<usize>(&out[1], 2).unwrap(), vec![2, 3]);
}

#[test]
fn errors_are_reported() {
    assert!(output("1 +", 1).is_err());
    assert!(output("1\n2", 1).unwrap_err().contains("expected 1 lines"));
    assert!(numbers::<f64>("1 2", 3).is_err());
}

#[test]
fn sections_cut_between_markers() {
    assert_eq!(section("a\n# start\nx\n# end\nb", "# start", "# end"), "# start\nx\n");
}

#[test]
fn a_program_imports_this_repos_libraries() {
    let out = microscope::run::output("\"nn:\" u_se< \"NN\"\nnn:r_elu -1.0 2.0\n", 1).unwrap();
    assert_eq!(out, vec!["0.0 2.0".to_string()]);
}

#[test]
fn a_program_imports_the_vendored_check_library() {
    let out = microscope::run::output("\"k:\" u_se< \"Check\"\n6 k:i_s 2 * 3\n", 1).unwrap();
    assert_eq!(out, vec!["ok".to_string()]);
}
