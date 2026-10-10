use embedding_explorer_web::micro::{head, program, run, words, CLUSTERS, NEAR, SOURCE};

fn near(c: &embedding_explorer_web::micro::Cloud, w: &str) -> Vec<&'static str> {
    let ws = words();
    let i = ws.iter().position(|x| *x == w).unwrap();
    c.near[i * NEAR..(i + 1) * NEAR].iter().map(|&j| ws[j - 1]).collect()
}

#[test]
fn words_of_a_kind_are_neighbors() {
    let c = run().unwrap();
    let animals = ["cat", "dog", "fox", "horse", "bird", "cow", "sheep", "mouse", "lion", "goat"];
    let numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
    // At least four of the five nearest are of the word's kind.
    assert!(near(&c, "cat").iter().filter(|w| animals.contains(w)).count() >= 4, "{:?}", near(&c, "cat"));
    assert!(near(&c, "three").iter().filter(|w| numbers.contains(w)).count() >= 4, "{:?}", near(&c, "three"));
    assert!(c.kept > 0.4 && c.kept < 1.0, "{}", c.kept);
}

#[test]
fn k_means_puts_each_kind_in_one_cluster() {
    let c = run().unwrap();
    let ws = words();
    assert!(c.cluster.iter().all(|&k| (1..=CLUSTERS).contains(&k)));
    for kind in [["red", "blue", "green", "brown", "black"], ["park", "river", "city", "home", "garden"]] {
        let ks: Vec<usize> = kind.iter().map(|w| c.cluster[ws.iter().position(|x| x == w).unwrap()]).collect();
        assert!(ks.iter().all(|&k| k == ks[0]), "{kind:?}: {ks:?}");
    }
}

#[test]
fn the_page_runs_the_programs_core() {
    assert!(SOURCE.contains(head()));
    assert!(microscope::source::rebound(&program()).is_empty(), "{:?}", microscope::source::rebound(&program()));
    assert_eq!(words().len(), 70);
}
