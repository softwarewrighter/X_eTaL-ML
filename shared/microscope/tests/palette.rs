//! The composer's model, natively: the palette's parts read from
//! palette.tsv, insertion at the caret, and composing (a run, its
//! LaTeX lines, its errors placed in the text, its expansion).

use microscope::palette::{compose, groups, insert, parts, Part};
use xetal_play::{decorate, Class};

#[test]
fn the_palette_has_its_groups_in_order_and_every_part_is_sound() {
    let parts = parts();
    assert_eq!(groups(&parts), ["Equation", "Pipeline", "Stages", "Imports"]);
    assert!(parts.len() >= 50, "{}", parts.len());
    for g in groups(&parts) {
        let labels: Vec<&str> = parts.iter().filter(|p| p.group == g).map(|p| p.label).collect();
        let mut unique = labels.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(labels.len(), unique.len(), "{g}: a label twice: {labels:?}");
    }
    for p in &parts {
        assert!(!p.label.is_empty() && !p.text.is_empty(), "{p:?}");
        assert!(p.back <= p.text.chars().count(), "{p:?}");
        assert!(!p.text.contains("\\n"), "{p:?}: a line break is a real one");
    }
    // A part that stands as X_eTaL (a pipeline's, a stage, an import) lexes
    // cleanly; an equation's parts are text inside a string.
    for p in parts.iter().filter(|p| p.group != "Equation") {
        assert!(!decorate(p.text).iter().any(|s| s.class == Class::Error), "{p:?} does not lex");
    }
}

#[test]
fn a_part_goes_in_at_the_caret_and_the_caret_lands_where_it_says() {
    let exp = Part { group: "Equation", label: "exp", text: "exp()", back: 1 };
    assert_eq!(insert("a + b", 4, 4, &exp), ("a + exp()b".to_string(), 8));
    // Over a selection, replacing it.
    assert_eq!(insert("a + b", 4, 5, &exp), ("a + exp()".to_string(), 8));
    // Past the end, at the end; a reversed selection as a forward one.
    assert_eq!(insert("ab", 9, 9, &exp), ("abexp()".to_string(), 6));
    assert_eq!(insert("abc", 2, 1, &exp), ("aexp()c".to_string(), 5));
    let stage = Part { group: "Pipeline", label: "|> stage", text: "\n|> ", back: 0 };
    assert_eq!(insert("X", 1, 1, &stage), ("X\n|> ".to_string(), 5));
}

#[test]
fn composing_runs_the_program_takes_its_latex_out_and_expands_it() {
    let src = "\"e:\" u_se< \"Equations\"\nW := f_loat 2 3 r_eshape 1 2 3 4 5 6\nx := f_loat 1 1 1\n\"$$\" c_at (\"z[i] | 2\" e:t_ex< \"sum j:=0..2 | W[i,j] * x[j]\") c_at \"$$\"\n\"z[i] | 2\" e:e_q< \"sum j:=0..2 | W[i,j] * x[j]\"\n";
    let c = compose(src);
    assert_eq!(c.err, "", "{c:?}");
    assert_eq!(c.tex, ["z_{i} = \\sum_{j=0}^{2} W_{i,j} \\, x_{j}"]);
    assert_eq!(c.out, ["6.0 15.0"]);
    let x = c.expansion.unwrap().unwrap();
    assert!(x.contains("grid") && !x.contains("e:e_q<"), "{x}");
}

#[test]
fn an_error_is_placed_in_the_text_and_a_refused_notation_names_its_rule() {
    let c = compose("x := 1\ny := x + \"a\"\n");
    assert!(c.err.contains("error[") && c.err.contains("--> ") && c.err.contains(":2:"), "{}", c.err);
    assert!(c.out.is_empty() && c.expansion.is_none());
    let c = compose("\"e:\" u_se< \"Equations\"\nv := 1 2 3\n\"t[k] | 3\" e:e_q< \" | v[k]2\"\n");
    assert!(c.err.contains("error[bad-equation]") && c.err.contains("v[k]2"), "{}", c.err);
}

#[test]
fn every_stage_of_the_palette_runs_in_a_pipeline_over_fitting_data() {
    // The data each stage's text names, as the demos name them.
    let head = "\"flow:\" u_se< \"Flow\"\n\"nn:\" u_se< \"NN\"\n\"nm:\" u_se< \"Norm\"\n\"cv:\" u_se< \"Conv\"\n\"at:\" u_se< \"Attention\"\nX := 2 3 r_eshape 1.0 2.0 3.0 4.0 5.0 7.0\nwb := 4 3 r_eshape 0.5\nb := 0.1 0.2 0.3\ng := 1.0 1.0 1.0\nK := 2 9 r_eshape 0.5\nB := 0.5 -0.25\nP := 1 4 4 r_eshape f_loat r_ange 16\nQ := 1 2 4 4 r_eshape f_loat r_ange 32\nWq := 3 2 r_eshape 0.5\nWk := 3 2 r_eshape 0.25\nM := at:c_ausal 2\nS := (Wq, Wk) at:s_cores X\n";
    for p in parts().into_iter().filter(|p| p.group == "Stages") {
        // A convolution stage takes pictures, pooling a batch of maps, the attention weights the scores, the others rows.
        let data = if p.text == "cv:p_ool" { "Q" } else if p.text.contains("cv:") { "P" } else if p.text.contains("at:w_eights") { "S" } else { "X" };
        let src = format!("{head}@ flow:p_ipe< c\"{data} |> {}\"\n", p.text);
        let c = compose(&src);
        assert_eq!(c.err, "", "{}: {}", p.label, c.err);
        assert!(!c.out.is_empty(), "{}: printed nothing", p.label);
    }
}
