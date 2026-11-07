//! The shell natively, built for this repository (no LESSON_* set): a
//! program run through microscope finds the libraries; the header and
//! footer name the site and the pins; the frames, the glossary, the
//! explanations and links of library functions; and the parts. Moved
//! from X_eTaL-fine-tuning (d78aca4) with its tests, those naming that
//! repository rewritten for this one's defaults.

use lesson::chrome::{build, pin, pins, Frame, FRAMES, REPO, SITE};
use microscope::libs::names;
use microscope::run::{output, printed};

/// Whether this is the default build (this repository's site, no
/// LESSON_* set): the checks of its defaults hold only then.
fn default_build() -> bool {
    option_env!("LESSON_SITE").is_none()
}

#[test]
fn a_program_imports_a_library_of_xetal_ml() {
    let out = output("\"lo:\" u_se< \"LoRA\"\n2 lo:p_arams 8 8\n", 1).unwrap();
    assert_eq!(out, ["32"]);
}

#[test]
fn the_adapted_layer_is_the_frozen_one_plus_the_correction() {
    // X_eTaL-fine-tuning's lora-hello layer: A x is 3.5, s is 2, so the adapter adds 7 B.
    let src = "\"lo:\" u_se< \"LoRA\"\n\
        A := 1 4 r_eshape 0.5 -0.5 1.0 0.25\n\
        B := 4 1 r_eshape 1.0 0.0 -1.0 0.5\n\
        (lo:d_elta (A, B, 2.0)) '+ '* i_nner 1.0 2.0 3.0 4.0\n";
    assert_eq!(output(src, 1).unwrap(), ["7.0 0.0 -7.0 3.5"]);
}

#[test]
fn a_general_library_microscope_embeds_is_found() {
    assert_eq!(output("\"k:\" u_se< \"Check\"\n6 k:i_s 2 * 3\n", 1).unwrap(), ["ok"]);
}

#[test]
fn an_error_is_reported_not_swallowed() {
    assert!(printed("1 +\n").is_err());
}

#[test]
fn the_header_and_footer_name_the_site_and_its_pins() {
    if !default_build() {
        return;
    }
    // Unset, LESSON_SITE is this repository's.
    assert_eq!(REPO, "https://github.com/softwarewrighter/X_eTaL-ML");
    assert_eq!((SITE.name, SITE.url, SITE.catalog), ("X_eTaL-ML", REPO, ("../", "All demos")));
    assert_eq!(SITE.commit, lesson::chrome::sha());
    // Unset, LESSON_PINS is this repository's two pins, in full.
    let names: Vec<&str> = pins().iter().map(|(n, _)| *n).collect();
    assert_eq!(names, ["X_eTaL-libraries", "X_eTaL"]);
    let file = |f: &str| std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(f)).unwrap().trim().to_string();
    assert_eq!(pins()[0].1, file("XETAL_LIBRARIES_COMMIT"));
    assert_eq!(pin("X_eTaL"), &file("XETAL_COMMIT")[..7]);
    assert_eq!(pin("no such"), "unknown");
    // Compact, no labels: host, short commit, yyyymmddThhmm.
    let b = build();
    let parts: Vec<&str> = b.split(' ').collect();
    assert_eq!(parts.len(), 3, "{b}");
    assert!(parts[1].len() == 7 && parts[1].chars().all(|c| c.is_ascii_hexdigit()), "{b}");
    let t = parts[2];
    assert!(t.len() == 13 && t.as_bytes()[8] == b'T' && t.chars().filter(|c| c.is_ascii_digit()).count() == 12, "{b}");
}

#[test]
fn the_six_frames_and_a_dependents_own() {
    let names: Vec<&str> = FRAMES.iter().map(|f| f.name()).collect();
    assert_eq!(names, ["question", "background", "rationale", "purpose", "summary", "next"]);
    assert_eq!(Frame::Other("proof").name(), "proof");
}

// New eyes (docs/new-eyes.md): ideas explained simply, library
// functions explained from their own source, code linked.

#[test]
fn the_glossary_is_well_formed() {
    let all = lesson::eli5::concepts();
    assert!(all.len() >= 13, "{}", all.len());
    let mut ids: Vec<&str> = all.iter().map(|c| c.id).collect();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), all.len(), "an id twice");
    for c in &all {
        assert!(!c.id.is_empty() && !c.term.is_empty(), "{c:?}");
        assert!(c.id.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-'), "{c:?}");
        // An explanation is a few plain sentences, not a line and not an essay.
        let words = c.simply.split_whitespace().count();
        assert!((15..=75).contains(&words), "{}: {words} words", c.id);
    }
    // Every line of the file is an idea or a comment: three fields, or
    // five with a worked example and the lesson that teaches it.
    for l in include_str!("../glossary.tsv").lines().filter(|l| !l.starts_with('#')) {
        assert!([3, 5].contains(&l.split('\t').count()), "{l}");
    }
    // The shared base names no lesson: that is a dependent's to say.
    assert!(!default_build() || all.iter().all(|c| c.lesson.is_empty()), "a base row names a lesson");
    // The worked example of a table times a list is X_eTaL's arithmetic.
    let times = all.iter().find(|c| c.id == "times").unwrap();
    assert!(times.example.contains("(12, 34)"), "{}", times.example);
    assert_eq!(output("(2 2 r_eshape 1 2 3 4) '+ '* i_nner 10 1\n", 1).unwrap(), ["12 34"]);
}

#[test]
fn marked_prose_is_taken_apart() {
    use lesson::eli5::{missing, pieces, used, Piece};
    let text = "A **bold** start, a [list|list] and [the table|table]; [plain] stays.";
    assert_eq!(
        pieces(text),
        [
            Piece::Text("A ".into()),
            Piece::Bold("bold".into()),
            Piece::Text(" start, a ".into()),
            Piece::Term { shown: "list".into(), id: "list".into() },
            Piece::Text(" and ".into()),
            Piece::Term { shown: "the table".into(), id: "table".into() },
            Piece::Text("; [plain] stays.".into()),
        ]
    );
    assert_eq!(used(&[text, "[again|list]"]), ["list", "table"]);
    assert_eq!(missing(&["[x|no-such-idea] [w|list]"]), ["no-such-idea"]);
}

#[test]
fn a_library_function_is_explained_from_its_own_source() {
    if !default_build() {
        return;
    }
    use lesson::explain::{calls, definition, home, imports, xref_page};
    let program = "\"lo:\" u_se< \"LoRA\"\n# lo:n_ot code, a comment\ny := (W, A, B, s) lo:f_orward x\nlo:r_ank lo:d_elta (A, B, s)\n";
    let ims = imports(program);
    assert_eq!((ims[0].alias.as_str(), ims[0].library.as_str(), ims[0].line.as_str()), ("lo:", "LoRA", "\"lo:\" u_se< \"LoRA\""));
    assert_eq!(calls(program), [("LoRA".to_string(), "f_orward".to_string()), ("LoRA".to_string(), "r_ank".to_string()), ("LoRA".to_string(), "d_elta".to_string())]);
    let d = definition("LoRA", "f_orward").unwrap();
    // The definition is a line of the library's file, at the line said.
    let source = microscope::libs::LIBRARIES.iter().find(|(f, _)| *f == "LoRA.xtl").unwrap().1;
    assert_eq!(source.lines().nth(d.line - 1).unwrap(), d.source);
    assert!(d.doc.starts_with("`(W, A, B, s) f_orward x`: the adapted layer"), "{}", d.doc);
    assert!(d.source_url.starts_with("https://github.com/softwarewrighter/X_eTaL-ML/blob/") && d.source_url.ends_with(&format!("/libs/LoRA/src/LoRA.xtl#L{}", d.line)), "{}", d.source_url);
    assert_eq!(d.xref, "../doc/libs-LoRA-src-LoRA.xtl.html#l.f_orward");
    // A function of several lines is taken whole.
    assert!(definition("LoRA", "r_ank").unwrap().source.lines().count() > 1);
    assert_eq!(definition("LoRA", "n_o_such"), None);
    // The site's own library is "this repository"'s; a general one is
    // found in its own repository, at the pinned commit.
    assert_eq!(home("LoRA"), "this repository");
    assert_eq!(home("Check"), "X_eTaL-libraries");
    assert_eq!(xref_page("Check").unwrap(), "../doc/work-libs-Check-src-Check.xtl.html");
    let pin = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../XETAL_LIBRARIES_COMMIT")).unwrap();
    assert!(definition("Check", "t_est").unwrap().source_url.contains(&format!("/X_eTaL-libraries/blob/{}/libs/Check/src/Check.xtl#L", pin.trim())));
}

#[test]
fn functions_in_code_link_to_the_cross_reference() {
    if !default_build() {
        return;
    }
    use lesson::code::{anchor, builtins, Linker};
    use xetal_play::Class;
    // As `xetal doc` names its anchors.
    assert_eq!(anchor("r_eshape"), "r_eshape");
    assert_eq!(anchor("r_/"), "r_-2f");
    assert_eq!(anchor("+"), "-2b");
    assert_eq!(anchor("s:i_f<"), "s.i_f-3c");
    let l = Linker::of("\"lo:\" u_se< \"LoRA\"\n");
    assert_eq!(l.href(Class::Builtin, "i_nner").unwrap(), "../doc/builtins.html#i_nner");
    assert_eq!(l.href(Class::Symbol, "*").unwrap(), "../doc/builtins.html#-2a");
    assert_eq!(l.href(Class::LibFunc, "lo:m_erge").unwrap(), "../doc/libs-LoRA-src-LoRA.xtl.html#l.m_erge");
    // Not functions: no link. A library the program did not import: none.
    assert_eq!(l.href(Class::Symbol, "("), None);
    assert_eq!(l.href(Class::Builtin, "'"), None);
    assert_eq!(l.href(Class::LibFunc, "zz:f_oo"), None);
    // Inside a library, l: is the library itself.
    assert_eq!(Linker::within("LoRA", "").href(Class::LibFunc, "l:d_elta").unwrap(), "../doc/libs-LoRA-src-LoRA.xtl.html#l.d_elta");
    assert_eq!(builtins("h := A '+ '* i_nner x\n4 4 r_eshape 'm_ax r_/ h\n"), ["i_nner", "r_eshape", "m_ax", "r_/"]);
}

#[test]
fn the_pages_run_the_libraries_of_the_pins() {
    if !default_build() {
        return;
    }
    // Built here, microscope embeds work/libs (X_eTaL-libraries at
    // XETAL_LIBRARIES_COMMIT): each embedded copy is the clone's.
    let clone = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../work/libs");
    let mut checked = 0;
    for (file, source) in microscope::libs::LIBRARIES {
        let path = clone.join(file.split('.').next().unwrap()).join("src").join(file);
        if path.is_file() {
            assert_eq!(*source, std::fs::read_to_string(path).unwrap(), "{file} differs from the pinned clone");
            checked += 1;
        }
    }
    assert!(checked >= 7, "only {checked} libraries of the clone are embedded");
    assert_eq!(output("\"e:\" u_se< \"Equations\"\nx := 1.0 2.0 3.0\n\"s[] | \" e:e_q< \"sum i:=0..2 | x[i]\"\n", 1).unwrap(), ["6.0"]);
}

// The teaching shell's parts (docs/plan.md, F5): presentation only.

#[test]
fn a_mode_shows_all_that_the_one_before_it_shows() {
    use lesson::mode::Mode;
    assert!(Mode::Beginner < Mode::Explorer && Mode::Explorer < Mode::Advanced);
    assert!(Mode::Beginner.shows(Mode::Beginner) && !Mode::Beginner.shows(Mode::Explorer) && !Mode::Beginner.shows(Mode::Advanced));
    assert!(Mode::Explorer.shows(Mode::Explorer) && !Mode::Explorer.shows(Mode::Advanced));
    assert!(Mode::ALL.iter().all(|m| Mode::Advanced.shows(*m)));
    let labels: Vec<&str> = Mode::ALL.iter().map(|m| m.label()).collect();
    assert_eq!(labels, ["Beginner", "Explorer", "Advanced"]);
    assert!(Mode::ALL.iter().all(|m| m.about().len() > 20));
}

#[test]
fn a_sum_is_written_out_from_the_numbers_given() {
    use lesson::arithmetic::{adds_up, commas, in_symbols, list, show, written};
    assert_eq!(show(3.5), "3.5");
    assert_eq!(show(7.0), "7");
    assert_eq!(show(-0.0000001), "0");
    assert_eq!(show(0.6442176872), "0.644218");
    assert_eq!(commas(16_777_216), "16,777,216");
    assert_eq!(commas(8), "8");
    assert_eq!(list(&[7.0, 0.0, -7.0, 3.5]), "7, 0, -7, 3.5");
    // It writes what it is given: the products and the total are X_eTaL's.
    assert_eq!(written("A x", &[0.5, -0.5], &[1.0, 2.0], &[0.5, -1.0], -0.5), "A x = 0.5 \u{00d7} 1 + (-0.5) \u{00d7} 2 = 0.5 + (-1) = -0.5");
    assert_eq!(in_symbols("y", "W", "x", 2, 3), "y(2) = W(2,1) \u{00d7} x(1) + W(2,2) \u{00d7} x(2) + W(2,3) \u{00d7} x(3)");
    assert!(adds_up(&[0.5, -1.0, 3.0, 1.0], 3.5, 1e-12) && !adds_up(&[0.5, -1.0], 3.5, 1e-12));
}

#[test]
fn a_change_is_said_from_its_measurements_without_a_verdict() {
    use lesson::narrate::{count, forbidden, held_out, moved};
    assert_eq!(moved("The loss", 0.184, 0.171), "The loss fell from 0.184 to 0.171");
    assert_eq!(moved("The loss", 0.1, 0.25), "The loss rose from 0.1 to 0.25");
    assert_eq!(moved("The loss", 0.1, 0.1), "The loss stayed at 0.1");
    assert_eq!(count("The numbers trained", 32, 64, Some((64, "the table itself"))), "The numbers trained went from 32 to 64, as many as the 64 of the table itself");
    assert_eq!(count("The numbers trained", 64, 128, Some((64, "the table itself"))), "The numbers trained went from 64 to 128, more than the 64 of the table itself");
    assert_eq!(count("The numbers trained", 16, 32, Some((64, "the table itself"))), "The numbers trained went from 16 to 32, fewer than the 64 of the table itself");
    assert_eq!(count("The numbers trained", 131_072, 131_072, None), "The numbers trained stayed at 131,072");
    let over = held_out((0.2, 0.05), (0.3, 0.4));
    assert!(over.contains("may be memorizing") && over.contains("fell from 0.2 to 0.05") && over.contains("rose from 0.3 to 0.4"), "{over}");
    let fine = held_out((0.2, 0.05), (0.3, 0.1));
    assert!(fine.contains("Both fell"), "{fine}");
    // No narration judges: every sentence these can make is free of the words.
    for text in [over, fine, held_out((0.2, 0.3), (0.3, 0.4)), held_out((0.2, 0.1), (0.3, 0.3)), moved("x", 1.0, 2.0), count("n", 1, 2, Some((2, "t")))] {
        assert_eq!(forbidden(&text), Vec::<&str>::new(), "{text}");
    }
    assert_eq!(forbidden("Higher rank is always better."), ["better", "always"]);
}

#[test]
fn a_prediction_is_answered_with_what_the_run_gives() {
    use lesson::predict::verdict;
    use yew::AttrValue;
    let choices = [AttrValue::from("y is unchanged"), AttrValue::from("y is zero")];
    assert_eq!(verdict(Some(0), 0, &choices), "You predicted: y is unchanged. The run bears that out.");
    assert_eq!(verdict(Some(1), 0, &choices), "You predicted: y is zero. The run gives: y is unchanged.");
    assert_eq!(verdict(None, 0, &choices), "The run gives: y is unchanged.");
}

#[test]
fn a_heat_maps_cells_legend_and_keys() {
    use lesson::heat::{fill, inspect, legend, scale_note, step, Matrix};
    let m = Matrix { rows: 2, cols: 3, items: vec![0.5, -0.25, 0.0, 1.0, 2.0, -3.0] };
    assert_eq!(inspect("W", &m, 0), "W, row 1, column 1: 0.5 (positive, so red)");
    assert_eq!(inspect("W", &m, 5), "W, row 2, column 3: -3 (negative, so blue)");
    assert_eq!(inspect("W", &m, 2), "W, row 1, column 3: 0 (zero, so white)");
    assert_eq!(inspect("x", &Matrix { rows: 3, cols: 1, items: vec![1.0, 2.0, 3.0] }, 1), "x, number 2: 2 (positive, so red)");
    assert_eq!(inspect("A", &Matrix { rows: 1, cols: 2, items: vec![1.0, -2.0] }, 1), "A, number 2: -2 (negative, so blue)");
    // The legend runs from -scale to +scale in the map's own colors.
    let l = legend(3.0);
    assert_eq!(l.iter().map(|(v, _)| *v).collect::<Vec<_>>(), [-3.0, -1.5, 0.0, 1.5, 3.0]);
    assert_eq!(l[4].1, fill(3.0, 3.0));
    assert_ne!(l[0].1, l[4].1);
    assert_ne!(scale_note(true), scale_note(false));
    // The arrow keys stay inside the grid.
    assert_eq!(step(0, 2, 3, "ArrowRight"), 1);
    assert_eq!(step(2, 2, 3, "ArrowRight"), 2);
    assert_eq!(step(1, 2, 3, "ArrowDown"), 4);
    assert_eq!(step(4, 2, 3, "ArrowDown"), 4);
    assert_eq!(step(4, 2, 3, "ArrowUp"), 1);
    assert_eq!(step(0, 2, 3, "ArrowLeft"), 0);
    assert_eq!(step(4, 2, 3, "Home"), 3);
    assert_eq!(step(3, 2, 3, "End"), 5);
    assert_eq!(step(3, 2, 3, "a"), 3);
}

#[test]
fn a_pipelines_run_is_its_trace_and_stepping_changes_nothing() {
    use lesson::stepper::{next, previous, trace};
    // X_eTaL-fine-tuning's lora-hello layer as a Flow pipeline; the trace is what
    // flow:s_tages< prints for it, from one run.
    let statement = "y := @ flow:p_ipe< c#\"\n   X $x\n|> \"h[k] | 1\" e:e_q< \"sum j:=0..3 | A[k,j] * x[j]\" $h\n|> \"u[i] | 4\" e:e_q< \"sum k:=0..0 | B[i,k] * h[k]\" $u\n|> \"y[i] | 4\" e:e_q< \"(sum j:=0..3 | W[i,j] * x[j]) + s * u[i]\"\"#";
    let pipe = &statement[statement.find("c#").unwrap()..];
    let program = format!(
        "\"e:\" u_se< \"Equations\"\n\"flow:\" u_se< \"Flow\"\nW := 4 4 r_eshape s_in 0.7 * f_loat r_ange 16\nA := 1 4 r_eshape 0.5 -0.5 1.0 0.25\nB := 4 1 r_eshape 1.0 0.0 -1.0 0.5\nX := 1.0 2.0 3.0 4.0\ns := 2.0\n@ flow:s_tages< {pipe}\n"
    );
    let line = output(&program, 1).unwrap().remove(0);
    let t = trace(&line, statement).unwrap();
    assert_eq!(t.stages.len(), 4);
    assert_eq!(t.stages.iter().map(|s| s.label.as_str()).collect::<Vec<_>>(), ["x", "h", "u", ""]);
    assert_eq!(t.stages[0].items, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(t.stages[1].items, [3.5]);
    assert_eq!(t.stages[2].items, [3.5, 0.0, -3.5, 1.75]);
    assert_eq!(t.stages[3].shape, [4]);
    // What each stage reads, by Flow's rule: h reads x; u reads h; y reads x and u.
    assert_eq!(t.stages.iter().map(|s| s.reads.clone()).collect::<Vec<_>>(), [vec![], vec![1], vec![2], vec![1, 3]]);
    assert!(t.stages[1].text.starts_with("\"h[k] | 1\" e:e_q<"), "{}", t.stages[1].text);
    // Stepping is a position in the record: it stays inside it, and the record is the same after.
    let before = t.clone();
    assert_eq!((next(0, 4), next(3, 4), previous(0), previous(3)), (1, 3, 0, 2));
    assert_eq!(t, before);
    // A record that does not match its pipeline is refused.
    assert!(trace("1;x;4;1.0;4.0;1.0 2.0 3.0 4.0", statement).is_err());
}

#[test]
fn a_macro_is_explained_and_linked_as_a_function_is() {
    if !default_build() {
        return;
    }
    use lesson::code::Linker;
    use lesson::explain::{calls, definition};
    use xetal_play::Class;
    // A macro library's export is `m:name< := ` in Name.xtlm.
    let program = "\"flow:\" u_se< \"Flow\"\n\"e:\" u_se< \"Equations\"\ny := @ flow:p_ipe< c\"X |> 2 * it\"\n";
    assert_eq!(calls(program), [("Flow".to_string(), "p_ipe".to_string())]);
    let d = definition("Flow", "p_ipe").unwrap();
    assert!(d.source.starts_with("m:p_ipe< := "), "{}", d.source);
    assert!(d.source_url.contains("/X_eTaL-ML/blob/") && d.source_url.contains("/libs/Flow/src/Flow.xtlm#L"), "{}", d.source_url);
    assert_eq!(d.xref, "../doc/libs-Flow-src-Flow.xtlm.html#m.p_ipe-3c");
    // Equations has functions and macros: each is found in its own file.
    assert!(definition("Equations", "e_q").unwrap().source_url.contains("/X_eTaL-libraries/blob/"));
    assert!(definition("Equations", "e_q").unwrap().xref.ends_with("Equations.xtlm.html#m.e_q-3c"));
    let l = Linker::of(program);
    assert_eq!(l.href(Class::Macro, "flow:p_ipe<").unwrap(), "../doc/libs-Flow-src-Flow.xtlm.html#m.p_ipe-3c");
    assert_eq!(l.href(Class::Macro, "e:e_q<").unwrap(), "../doc/work-libs-Equations-src-Equations.xtlm.html#m.e_q-3c");
}

#[test]
fn a_state_named_in_an_address_is_read() {
    use lesson::link::value;
    assert_eq!(value("?stage=3&rank=5&mode=advanced", "rank").as_deref(), Some("5"));
    assert_eq!(value("stage=3", "stage").as_deref(), Some("3"));
    assert_eq!(value("?stage=3", "rank"), None);
    assert_eq!(value("", "rank"), None);
    // Outside the browser there is no address: a page starts as it always did.
    assert_eq!(lesson::link::number("stage", 1, 4), None);
    assert_eq!(lesson::mode::Mode::from_link(), None);
}

#[test]
fn a_dependent_names_its_site_glossary_and_pins() {
    // Built as a dependent builds it (LESSON_ROOT and the rest set; see
    // build.rs): its site, its glossary's rows after the base, its pins,
    // its own libraries "this repository"'s and X_eTaL-ML's its own.
    // Checked by hand with X_eTaL-fine-tuning as the dependent; the
    // default build sets none.
    let Some(site) = option_env!("LESSON_SITE") else { return };
    let name = site.split('|').next().unwrap();
    assert_eq!(SITE.name, name);
    assert_ne!(SITE.name, "X_eTaL-ML");
    if let Some(pins_named) = option_env!("LESSON_PINS") {
        let names: Vec<&str> = pins().iter().map(|(n, _)| *n).collect();
        let want: Vec<&str> = pins_named.split(':').map(|p| p.split('=').next().unwrap()).collect();
        assert_eq!(names, want);
        assert!(pins().iter().all(|(_, c)| c.len() == 40), "{:?}", pins());
    }
    if option_env!("LESSON_GLOSSARY").is_some() {
        let all = lesson::eli5::concepts();
        assert!(all.len() > 13 && all.iter().any(|c| !c.lesson.is_empty()), "the dependent's rows are read");
    }
    assert_eq!(lesson::explain::home("LoRA"), "X_eTaL-ML");
    assert!(lesson::explain::xref_page("LoRA").unwrap().starts_with("../doc/"), "{:?}", lesson::explain::xref_page("LoRA"));
}
