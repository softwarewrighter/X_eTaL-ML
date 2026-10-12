//! Every demo, run through xetal-play with the libraries' store (as the
//! browser runs it, seed 1, ASCII frames), prints what its reg-rs
//! baseline recorded; every library's types are what the page shows.

use xetal_ml_libraries::{store, GROUPS, LIBRARIES};

#[test]
fn every_demo_runs_as_recorded() {
    store::install();
    xetal_grid::set_ascii(true);
    let mut n = 0;
    for lib in LIBRARIES {
        assert!(!lib.demos.is_empty(), "{} has no demo", lib.name);
        for demo in lib.demos {
            let run = xetal_play::run(demo.source, 1);
            assert_eq!(run.err, "", "{}/{}: error", lib.name, demo.name);
            // The command line reports each picture it writes ("drawn ..."); the browser shows it instead.
            let expected: String = demo.expected.lines().filter(|l| !l.starts_with("drawn ")).map(|l| format!("{l}\n")).collect();
            assert_eq!(run.out, expected, "{}/{}: output", lib.name, demo.name);
            n += 1;
        }
    }
    assert!(n >= LIBRARIES.len());
}

#[test]
fn every_library_checks_to_its_pinned_types() {
    store::install();
    for lib in LIBRARIES {
        // A macro-only library (Tags) pins its macros' types.
        let code = if lib.source.is_empty() { lib.macros } else { lib.source };
        let lines = xetal_play::check(code);
        assert_eq!(lines.join("\n") + "\n", lib.types, "{} types", lib.name);
    }
}

#[test]
fn the_catalog_has_every_library() {
    assert!(LIBRARIES.len() >= 8);
    for lib in LIBRARIES {
        assert!(!lib.alias.is_empty() && lib.alias.ends_with(':'), "{} alias", lib.name);
        assert!(lib.docs.contains("<table>"), "{} docs", lib.name);
    }
}

#[test]
fn pages_show_xetal_rendered_not_typed() {
    // Typed spellings that the rendered form always decorates.
    for lib in LIBRARIES {
        for typed in ["u_se&lt;", "r_/", " := ", "s_elect", "t_ally"] {
            assert!(!lib.docs.contains(typed), "{}: typed {typed:?} on its reference page", lib.name);
        }
        assert!(lib.docs.contains("class=\"c-"), "{}: nothing rendered", lib.name);
    }
}

#[test]
fn every_library_is_in_one_group() {
    for lib in LIBRARIES {
        let n = GROUPS.iter().filter(|(_, names)| names.contains(&lib.name)).count();
        assert_eq!(n, 1, "{} is in {n} groups (site/src/lib.rs GROUPS)", lib.name);
    }
    for (_, names) in GROUPS {
        for name in *names {
            assert!(xetal_ml_libraries::library(name).is_some(), "group names {name}, no such library");
        }
    }
}

#[test]
fn expansions_of_system_macros() {
    use xetal_ml_libraries::expansion;
    store::install();
    assert_eq!(expansion("1 + 2"), None);
    assert_eq!(expansion("\"t:\" u_se< \"Strings\"\nt:u_pper \"a\""), None);
    assert_eq!(
        expansion("\"n = 0\" i_f< \"0.0; 100 / n\""),
        Some(Ok("{ @ -> (n = 0) ? 0.0; 100 / n } @".to_string()))
    );
    assert!(matches!(expansion("\"c\" i_f< \"1\""), Some(Err(_))));
}

#[test]
fn expansion_marks_the_lines_a_macro_became() {
    use xetal_ml_libraries::expansion_marked;
    store::install();
    let src = "n := 4\n\"n = 0\" i_f< \"0.0; 100 / n\"\nn + 1\n";
    let lines = expansion_marked(src).unwrap().unwrap();
    let changed: Vec<&str> = lines.iter().filter(|(_, c)| *c).map(|(l, _)| l.as_str()).collect();
    assert_eq!(changed, vec!["{ @ -> (n = 0) ? 0.0; 100 / n } @"]);
    assert_eq!(lines.len(), 3);
    assert!(expansion_marked("1 + 2").is_none());
}

#[test]
fn library_macros_run_and_expand_in_the_browser_store() {
    use xetal_ml_libraries::{expansion, library};
    store::install();
    assert!(!library("Net").unwrap().macros.is_empty(), "Net has its .xtlm built in");
    let src = "\"net:\" u_se< \"Net\"\n\"2 16 relu 16 relu 3 softmax\" net:p_arams< @";
    assert_eq!(xetal_play::run(src, 1).out, "371\n");
    let e = expansion(src).unwrap().unwrap();
    assert!(e.contains("371"), "{e}");
    assert!(xetal_play::run("\"net:\" u_se< \"Net\"\n\"2 bogus 3\" net:p_arams< @", 1).err.contains("not a size or an activation"));
}
