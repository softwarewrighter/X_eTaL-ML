//! The timeline's events, scale, layout and navigation, natively.

use ml_timeline::{
    Demo, Event, EMPTY, demo, fill, undemoed, FIRST, FOLD, Go, Kind, LEAD, MIN_YEAR, NOW, Scale, Side, check, events, frame, hash,
    key, mini, mini_year, parse, place, scroll_for, sources, target, unsourced,
};

const DATA: &str = include_str!("../../../data/timeline.json");

/// Every demo's `demo.toml`, read from demos/.
fn all_demos() -> Vec<Demo> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../demos");
    let mut found: Vec<Demo> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|d| std::fs::read_to_string(d.ok()?.path().join("demo.toml")).ok())
        .filter_map(|t| demo(&t))
        .collect();
    found.sort_by(|a, b| a.slug.cmp(&b.slug));
    found
}

/// The timeline's events, the demos' filled from their demo.toml, as the
/// page shows them.
fn filled() -> Vec<Event> {
    let mut list = events(DATA).unwrap();
    assert_eq!(fill(&mut list, &all_demos()), Vec::<String>::new());
    list
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// The scale of the timeline's own events.
fn ours() -> Scale {
    let list = filled();
    let sized: Vec<(f64, f64)> = list.iter().map(|e| (e.at(), e.size(false).0)).collect();
    Scale::new(FIRST, NOW, &sized, 260.0)
}

#[test]
fn the_events_read_and_check() {
    let list = filled();
    assert!(list.len() >= 12, "a few events to show the strip");
    assert_eq!(check(&list), Vec::<String>::new());
    assert!(list.windows(2).all(|w| w[0].at() <= w[1].at()), "in time order");
    assert_eq!(list[0].year, FIRST);
}

#[test]
fn an_event_stands_at_its_month_or_mid_year() {
    let json = r#"[{"year": 2017, "month": 6, "when": "June 2017", "title": "T", "kind": "note"},
                   {"year": 1956, "when": "1956", "title": "D", "kind": "milestone"}]"#;
    let list = events(json).unwrap();
    assert_eq!(list[0].year, 1956, "sorted");
    assert!(near(list[0].at(), 1956.5));
    assert!(near(list[1].at(), 2017.0 + 5.5 / 12.0));
}

#[test]
fn check_finds_what_is_wrong() {
    let json = r#"[{"year": 1930, "when": "1930", "title": "Early", "kind": "note"},
                   {"year": 2000, "month": 13, "when": "2000", "title": "M", "kind": "note"},
                   {"year": 2001, "when": "1999", "title": "W", "kind": "note"},
                   {"year": 2002, "when": "2002", "title": "D", "kind": "demo"},
                   {"year": 2003, "when": "2003", "title": "S", "kind": "note", "demo": "cnn-digits"},
                   {"year": 2004, "when": "2004", "title": "P", "kind": "paper"},
                   {"year": 2005, "when": "2005", "title": " ", "kind": "note"}]"#;
    let wrong = check(&events(json).unwrap());
    let says = |what: &str| wrong.iter().filter(|w| w.contains(what)).count();
    assert_eq!(says("outside"), 1);
    assert_eq!(says("not 1 to 12"), 1);
    assert_eq!(says("does not name the year"), 1);
    assert_eq!(says("month alone"), 1, "month 13 is named by no when");
    assert_eq!(says("slug"), 2);
    assert_eq!(says("the idea it shows"), 1);
    assert_eq!(says("link"), 1);
    assert_eq!(says("no title"), 1);
    assert_eq!(wrong.len(), 9);
}

#[test]
fn kinds_and_topics_read() {
    let json = r#"[{"year": 1989, "when": "1989", "title": "CNN", "kind": "demo", "demo": "cnn-digits",
                    "topics": ["CNN"], "image": "shots/cnn-digits.webp", "side": "below"}]"#;
    let e = &events(json).unwrap()[0];
    assert_eq!((e.kind, e.demo.as_deref(), e.side), (Kind::Demo, Some("cnn-digits"), Some(Side::Below)));
    assert_eq!(e.topics, vec!["CNN".to_string()]);
    assert!(e.size(false).0 > 290.0, "a picture's box is wider");
    assert!(e.size(true).0 < e.size(false).0, "narrower on a phone");
}

#[test]
fn a_year_is_as_wide_as_its_callouts_need() {
    let one = Scale::new(2000, 2002, &[(2000.5, 230.0)], 0.0);
    let many = Scale::new(2000, 2002, &[(2000.5, 230.0); 9], 0.0);
    assert!(near(one.segments[0].width, MIN_YEAR.max(123.0 + 16.0)));
    assert!(near(many.segments[0].width, 9.0 * 123.0 + 16.0), "half of each box, alternating sides");
    assert_eq!(many.segments[0].events, 9);
    assert!(near(many.segments[1].width, EMPTY), "a lone empty year is narrow");
}

#[test]
fn runs_of_empty_years_fold() {
    // 2000 has an event; 2001-2002 are two empty years; 2003; 2004-2010 seven.
    let s = Scale::new(2000, 2011, &[(2000.5, 100.0), (2003.5, 100.0), (2011.2, 100.0)], 10.0);
    let spans: Vec<(i32, i32)> = s.segments.iter().map(|g| (g.from, g.to)).collect();
    assert_eq!(spans, vec![(2000, 2001), (2001, 2002), (2002, 2003), (2003, 2004), (2004, 2011), (2011, 2012)]);
    assert!(s.segments[4].folded() && !s.segments[1].folded());
    assert!(near(s.segments[4].width, FOLD));
    assert!(near(s.segments[0].left, 10.0), "the margin first");
    assert!(s.segments.windows(2).all(|w| near(w[0].right(), w[1].left)), "edge to edge");
    assert!(near(s.width(), s.segments[5].right() + 10.0));
}

#[test]
fn x_and_year_at_are_inverse() {
    let s = ours();
    for at in [1943.0, 1943.95, 1950.75, 1961.3, 1986.4, 2017.42, 2026.99] {
        assert!(near(s.year_at(s.x(at)), at), "{at}");
    }
    assert!(near(s.x(1900.0), s.x(f64::from(FIRST))), "held at the start");
    assert!(near(s.year_at(1e9), f64::from(NOW + 1)), "held at the end");
    let xs: Vec<f64> = (FIRST..=NOW).map(|y| s.x(f64::from(y))).collect();
    assert!(xs.windows(2).all(|w| w[0] < w[1]), "rising");
}

#[test]
fn our_quiet_years_fold_and_busy_ones_are_wide() {
    let s = ours();
    let fold = s.segment(1977).unwrap();
    assert!(fold.folded() && fold.from == 1975 && fold.to == 1980, "1975-1979 is one break");
    assert!(s.segment(2023).unwrap().width > 2.0 * s.segment(1989).unwrap().width, "a busy year is wide");
    assert!(s.segments.iter().all(|g| g.events == 0 || g.width >= MIN_YEAR));
}

#[test]
fn the_layout_keeps_order_and_clearance() {
    let list = events(DATA).unwrap();
    let s = ours();
    let items: Vec<_> = list.iter().map(|e| (e.at(), e.size(false), e.side)).collect();
    let placed = place(&items, &s);
    assert_eq!(placed.len(), list.len());
    for side in [Side::Above, Side::Below] {
        let on: Vec<_> = placed.iter().filter(|p| p.side == side).collect();
        assert!(on.windows(2).all(|w| w[0].left + w[0].width <= w[1].left + 1e-6), "{side:?} clear");
    }
    assert!(placed.windows(2).all(|w| w[0].side != w[1].side), "alternating");
    assert!(placed.iter().zip(&list).all(|(p, e)| near(p.anchor, s.x(e.at()))));
}

#[test]
fn the_buttons_and_keys_move_the_view() {
    let s = ours();
    let (view, most) = (1000.0, s.width() - 1000.0);
    assert!(near(scroll_for(Go::Start, &s, 500.0, view), 0.0));
    assert!(near(scroll_for(Go::End, &s, 500.0, view), most));
    assert!(near(scroll_for(Go::On, &s, 500.0, view), 1300.0));
    assert!(near(scroll_for(Go::Back, &s, 500.0, view), 0.0), "held at the start");
    assert!(near(scroll_for(Go::On, &s, most - 10.0, view), most), "held at the end");
    assert!(near(scroll_for(Go::Right, &s, 500.0, view), 700.0));
    assert!(near(scroll_for(Go::To(1989.0), &s, 0.0, view), s.x(1989.0) - LEAD));
    assert!(near(scroll_for(Go::Center(2017.5), &s, 0.0, view), s.x(2017.5) - 500.0));
    assert_eq!(key("Home"), Some(Go::Start));
    assert_eq!(key("End"), Some(Go::End));
    assert_eq!(key("PageDown"), Some(Go::On));
    assert_eq!(key("ArrowLeft"), Some(Go::Left));
    assert_eq!(key("a"), None);
}

#[test]
fn the_hash_and_year_box_parse() {
    assert_eq!(parse("#1989"), Some((1989, None)));
    assert_eq!(parse(" 1989 "), Some((1989, None)));
    assert_eq!(parse("#2017-06"), Some((2017, Some(6))));
    assert_eq!(parse("2017-6"), Some((2017, Some(6))));
    assert_eq!(parse("1990s"), Some((1990, None)));
    for bad in ["", "#", "89", "19890", "2017-13", "2017-0", "2017-", "2017-006", "abcd", "2017-x1"] {
        assert_eq!(parse(bad), None, "{bad}");
    }
    assert!(near(target((2017, Some(6))), 2017.0 + 5.0 / 12.0));
    assert!(near(target((1989, None)), 1989.0));
}

#[test]
fn the_hash_follows_the_view() {
    let s = ours();
    // A narrow view, so even 2023 opens with its month at the lead.
    let at = |t: f64| scroll_for(Go::To(t), &s, 0.0, 300.0);
    assert_eq!(hash(&s, at(1989.0)), "#1989");
    assert_eq!(hash(&s, at(target((2017, Some(6))))), "#2017-06");
    assert_eq!(hash(&s, at(target((1989, Some(12))))), "#1989-12");
    assert_eq!(hash(&s, at(1975.5)), "#1975", "inside a fold, the year alone");
    assert_eq!(hash(&s, at(1951.0)), "#1951", "an empty year, the year alone");
    for text in ["#1958", "#1997-05", "#2023-10", "#2020-06"] {
        let back = hash(&s, at(target(parse(text).unwrap())));
        assert_eq!(back, text, "round trip");
    }
}

#[test]
fn the_minimap_spans_every_year_alike() {
    let s = ours();
    assert!(near(mini(&s, f64::from(FIRST)), 0.0));
    assert!(near(mini(&s, f64::from(NOW + 1)), 1.0));
    let span = f64::from(NOW + 1 - FIRST);
    assert!(near(mini(&s, 1975.0) - mini(&s, 1974.0), 1.0 / span), "a folded year as wide as any");
    assert!(near(mini_year(&s, mini(&s, 2017.3)), 2017.3));
    let (from, to) = frame(&s, s.x(1989.0), s.x(1998.0) - s.x(1989.0));
    assert!(near(from, 1989.0) && near(to, 1998.0));
}

#[test]
fn a_when_names_its_month_alone() {
    let json = r#"[{"year": 2017, "month": 6, "when": "June 2017", "title": "Fine", "kind": "note"},
                   {"year": 2017, "month": 6, "when": "2017", "title": "Unnamed", "kind": "note"},
                   {"year": 2017, "month": 6, "when": "July 2017", "title": "Other", "kind": "note"},
                   {"year": 2017, "month": 6, "when": "June-July 2017", "title": "Two", "kind": "note"},
                   {"year": 1956, "when": "Summer 1956", "title": "Season", "kind": "note"},
                   {"year": 1956, "when": "May 1956", "title": "Unsaid", "kind": "note"}]"#;
    let wrong = check(&events(json).unwrap());
    let named = |t: &str| wrong.iter().any(|w| w.contains(&format!("{t:?}")));
    assert_eq!(wrong.len(), 4, "{wrong:?}");
    assert!(named("Unnamed") && named("Other") && named("Two") && named("Unsaid"));
}

#[test]
fn sources_read_from_their_table() {
    let md = "# Sources\n\n| Year | Month | Title | Checked | Link |\n| ---- | ----- | ----- | ------- | ---- |\n\
              | 1958 | 11 | The perceptron | Rosenblatt, Psychological Review 65 (6) | <https://doi.org/10.1037/h0042519> |\n\
              | 1956 | - | The Dartmouth workshop | a reference page | <https://en.wikipedia.org/wiki/Dartmouth_workshop> |\n\
              | not | a | row |\n";
    let rows = sources(md);
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[0].year, rows[0].month, rows[0].title.as_str()), (1958, Some(11), "The perceptron"));
    assert_eq!(rows[0].link, "https://doi.org/10.1037/h0042519");
    assert_eq!(rows[1].month, None);
}

#[test]
fn unsourced_finds_what_is_missing_or_stale() {
    let json = r#"[{"year": 1958, "month": 11, "when": "November 1958", "title": "Sourced", "kind": "paper", "href": "https://a"},
                   {"year": 1959, "when": "1959", "title": "Missing", "kind": "note"},
                   {"year": 1960, "when": "1960", "title": "Elsewhere", "kind": "paper", "href": "https://b"},
                   {"year": 1961, "when": "1961", "title": "Ours", "kind": "demo", "demo": "k-means"}]"#;
    let md = "| 1958 | 11 | Sourced | the paper | <https://a> |\n| 1960 | - | Elsewhere | the paper | <https://c> |\n\
              | 1962 | - | Orphan | a page | <https://d> |\n| 1958 | 11 | Sourced | again | http://a |\n";
    let wrong = unsourced(&events(json).unwrap(), &sources(md));
    let says = |what: &str| wrong.iter().filter(|w| w.contains(what)).count();
    assert_eq!(says("no source"), 1, "{wrong:?}");
    assert_eq!(says("its source is"), 1);
    assert_eq!(says("no event"), 1);
    assert_eq!(says("twice"), 1);
    assert_eq!(says("an https link"), 1);
    assert_eq!(wrong.len(), 5, "a demo needs no source: {wrong:?}");
}

#[test]
fn every_event_has_its_source() {
    let rows = sources(include_str!("../../../../docs/ml-timeline-sources.md"));
    assert_eq!(unsourced(&events(DATA).unwrap(), &rows), Vec::<String>::new());
}

#[test]
fn a_demo_reads_from_its_toml() {
    let toml = "# A demo.\nslug = \"net-macro\"\ntitle = \"Network macro\"\nsummary = \"A network as one line: a spec such as \\\"2 16 relu\\\" becomes code. More.\"\n\
                status = \"live\"\norder = 3\n[web]\ntitle = \"not this\"\n";
    let d = demo(toml).unwrap();
    assert_eq!((d.slug.as_str(), d.title.as_str(), d.live()), ("net-macro", "Network macro", true));
    assert_eq!(d.summary, "A network as one line: a spec such as \"2 16 relu\" becomes code. More.");
    assert_eq!(d.short(), "A network as one line.");
    assert_eq!(demo("title = \"no slug\"\n"), None);
}

#[test]
fn fill_takes_a_demo_from_its_toml() {
    let demos = [
        Demo { slug: "k-means".into(), title: "k-means".into(), summary: "Clustering. Then more.".into(), status: "live".into() },
        Demo { slug: "draft".into(), status: "draft".into(), ..Demo::default() },
    ];
    let json = r#"[{"year": 1957, "when": "1957", "kind": "demo", "demo": "k-means", "shows": "Lloyd's algorithm"},
                   {"year": 1958, "when": "1958", "kind": "demo", "demo": "draft", "shows": "x"},
                   {"year": 1959, "when": "1959", "kind": "demo", "demo": "gone", "shows": "x"},
                   {"year": 1960, "when": "1960", "title": "Twice", "kind": "demo", "demo": "k-means", "shows": "x"}]"#;
    let mut list = events(json).unwrap();
    let wrong = fill(&mut list, &demos);
    let e = &list[0];
    assert_eq!((e.title.as_str(), e.text.as_str()), ("k-means", "Clustering."));
    assert_eq!((e.href.as_deref(), e.image.as_deref()), (Some("../k-means/"), Some("shots/k-means.webp")));
    assert_eq!(e.tip.as_deref(), Some("Clustering. Then more."));
    assert!(e.size(false).0 > 290.0, "a pictured box");
    let says = |what: &str| wrong.iter().filter(|w| w.contains(what)).count();
    assert_eq!((says("not live"), says("no such demo"), says("come from its demo.toml")), (1, 1, 1), "{wrong:?}");
    assert_eq!(wrong.len(), 3);
    let missing = undemoed(&list[..1], &demos);
    assert_eq!(missing, Vec::<String>::new(), "a draft needs no event");
    assert_eq!(undemoed(&list, &demos), vec!["demo k-means: on the timeline 2 times".to_string()]);
    assert_eq!(undemoed(&[], &demos), vec!["demo k-means: live, but not on the timeline".to_string()]);
}

#[test]
fn every_live_demo_is_on_the_timeline_once() {
    let demos = all_demos();
    assert!(demos.iter().filter(|d| d.live()).count() >= 18, "the live demos read");
    assert_eq!(undemoed(&filled(), &demos), Vec::<String>::new());
}
