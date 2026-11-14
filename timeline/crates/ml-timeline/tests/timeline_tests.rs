//! The timeline's events, scale, layout and navigation, natively.

use ml_timeline::{
    EMPTY, FIRST, FOLD, Go, Kind, LEAD, MIN_YEAR, NOW, Scale, Side, check, events, frame, hash,
    key, mini, mini_year, parse, place, scroll_for, target,
};

const DATA: &str = include_str!("../../../data/timeline.json");

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

/// The scale of the timeline's own events.
fn ours() -> Scale {
    let list = events(DATA).unwrap();
    let sized: Vec<(f64, f64)> = list.iter().map(|e| (e.at(), e.size(false).0)).collect();
    Scale::new(FIRST, NOW, &sized, 260.0)
}

#[test]
fn the_events_read_and_check() {
    let list = events(DATA).unwrap();
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
    assert_eq!(says("month"), 1);
    assert_eq!(says("does not name"), 1);
    assert_eq!(says("slug"), 2);
    assert_eq!(says("link"), 1);
    assert_eq!(says("no title"), 1);
    assert_eq!(wrong.len(), 7);
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
    let fold = s.segment(1975).unwrap();
    assert!(fold.folded() && fold.from == 1970 && fold.to == 1986, "1970-1985 is one break");
    assert!(s.segment(1997).unwrap().width > s.segment(1989).unwrap().width, "two events need more");
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
