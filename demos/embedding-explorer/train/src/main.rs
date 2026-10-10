//! Writes the embedding explorer's corpus into `../data/`: 4,000
//! sentences made from templates over six kinds of words (animals,
//! foods, colors, numbers, places, actions) with a fixed seed, so that
//! words of a kind turn up in the same company; within a kind, each
//! word has its habits (an animal's favorite food, place and action, a
//! food's usual color, counting mostly in order), so the words of a
//! kind are alike but not the same. `words.txt` holds the
//! vocabulary, a word a line; `ids.txt` the corpus as word numbers
//! (from 1), each sentence ended by a 0; `kinds.txt` each word's kind
//! (for the tests); `sample.txt` the first 20
//! sentences as text, to read. The embeddings themselves are made by
//! the demo, in X_eTaL, from which words occur near which.

const ANIMALS: [&str; 10] = ["cat", "dog", "fox", "horse", "bird", "cow", "sheep", "mouse", "lion", "goat"];
const FOODS: [&str; 10] = ["bread", "apple", "cheese", "fish", "rice", "soup", "cake", "milk", "egg", "plum"];
const COLORS: [&str; 10] = ["red", "blue", "green", "brown", "black", "white", "yellow", "gray", "pink", "purple"];
const NUMBERS: [&str; 10] = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const PLACES: [&str; 10] = ["park", "river", "city", "home", "garden", "forest", "school", "market", "beach", "farm"];
const ACTIONS: [&str; 8] = ["runs", "walks", "sleeps", "jumps", "sits", "swims", "hides", "plays"];
const EATS: [&str; 3] = ["eats", "likes", "wants"];

/// A small linear congruential generator: the same corpus every run.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
    fn pick<'a>(&mut self, xs: &[&'a str]) -> &'a str {
        xs[self.next(xs.len())]
    }
}

/// Mostly (six times in ten) the word's habit, else any of the kind.
fn habit<'a>(r: &mut Lcg, xs: &[&'a str], usual: usize) -> &'a str {
    if r.next(10) < 6 { xs[usual % xs.len()] } else { r.pick(xs) }
}

fn sentence(r: &mut Lcg) -> Vec<String> {
    let a = r.next(ANIMALS.len());
    let f = r.next(FOODS.len());
    let s: Vec<&str> = match r.next(6) {
        0 => vec!["the", r.pick(&COLORS), ANIMALS[a], r.pick(&EATS), "the", habit(r, &FOODS, 3 * a + 1)],
        1 => vec!["the", ANIMALS[a], habit(r, &ACTIONS, a), "in", "the", habit(r, &PLACES, a)],
        2 => vec!["the", FOODS[f], "is", habit(r, &COLORS, 7 * f + 2)],
        3 => {
            let n = r.next(8);
            match r.next(10) < 6 {
                true => vec!["we", "count", NUMBERS[n], NUMBERS[n + 1], NUMBERS[n + 2]],
                false => vec!["we", "count", r.pick(&NUMBERS), r.pick(&NUMBERS), r.pick(&NUMBERS)],
            }
        }
        4 => vec!["we", "go", "to", "the", r.pick(&PLACES), "at", r.pick(&NUMBERS)],
        _ => vec!["a", r.pick(&COLORS), ANIMALS[a], habit(r, &ACTIONS, a), "at", "the", habit(r, &PLACES, a)],
    };
    s.into_iter().map(String::from).collect()
}

/// Each word's kind: 1 animals, 2 foods, 3 colors, 4 numbers, 5
/// places, 6 actions, 0 the little words (the, a, eats, count, ...).
fn kind(w: &str) -> usize {
    [&ANIMALS[..], &FOODS[..], &COLORS[..], &NUMBERS[..], &PLACES[..], &ACTIONS[..]].iter().position(|k| k.contains(&w)).map_or(0, |i| i + 1)
}

/// Usage: embedding-explorer-train [SENTENCES [DIR]] (default 4000 into ../data).
fn main() {
    let here = env!("CARGO_MANIFEST_DIR");
    let args: Vec<String> = std::env::args().collect();
    let count: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(4000);
    let data = args.get(2).cloned().unwrap_or(format!("{here}/../data"));
    let mut r = Lcg(1);
    let sentences: Vec<Vec<String>> = (0..count).map(|_| sentence(&mut r)).collect();
    // The vocabulary in order of first appearance.
    let mut words: Vec<String> = vec![];
    for w in sentences.iter().flatten() {
        if !words.contains(w) {
            words.push(w.clone());
        }
    }
    let id = |w: &String| words.iter().position(|v| v == w).unwrap() + 1;
    let ids: Vec<String> = sentences.iter().flat_map(|s| s.iter().map(|w| id(w).to_string()).chain(["0".to_string()])).collect();
    std::fs::create_dir_all(&data).expect("data/");
    std::fs::write(format!("{data}/words.txt"), words.join("\n") + "\n").expect("words.txt");
    std::fs::write(format!("{data}/ids.txt"), ids.join(" ") + "\n").expect("ids.txt");
    std::fs::write(format!("{data}/kinds.txt"), words.iter().map(|w| kind(w).to_string()).collect::<Vec<_>>().join(" ") + "\n").expect("kinds.txt");
    std::fs::write(format!("{data}/sample.txt"), sentences.iter().take(20).map(|s| s.join(" ")).collect::<Vec<_>>().join("\n") + "\n").expect("sample.txt");
    eprintln!("wrote {} words, {} tokens in {} sentences", words.len(), ids.len(), sentences.len());
}
