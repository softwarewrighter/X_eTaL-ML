//! The model: the demo's own X_eTaL program (world-model.xtl): the
//! world's step, the frames, the network's prediction and the likeliest
//! place in it, run one step at a time from the three ball positions
//! and the true state the page holds. Nothing here knows about the
//! browser.

use microscope::run::{numbers, output_with, section};

/// The command-line program.
pub const SOURCE: &str = include_str!("../../world-model.xtl");

/// The weights and the held-out states, as the page's X_eTaL finds them.
pub const DATA: [(&str, &str); 5] = [
    ("data/w1.txt", include_str!("../../data/w1.txt")),
    ("data/b1.txt", include_str!("../../data/b1.txt")),
    ("data/w2.txt", include_str!("../../data/w2.txt")),
    ("data/b2.txt", include_str!("../../data/b2.txt")),
    ("data/held-out.txt", include_str!("../../data/held-out.txt")),
];

/// The box's side in cells; the ball is 2 x 2, so a position is 0 to 10.
pub const SIDE: usize = 12;
pub const TOP: i64 = 10;
pub const SPEEDS: [i64; 4] = [-2, -1, 1, 2];

/// A state of the world: x, y, vx, vy.
pub type State = [i64; 4];
/// The ball's place, x y.
pub type Place = (i64, i64);

/// The program's head: the world, the network, the likeliest place.
pub fn head() -> &'static str {
    section(SOURCE, "\"nn:\" u_se< \"NN\"", "# -- end of the core")
}

/// The states the network never trained on (data/held-out.txt).
pub fn held_out() -> Vec<State> {
    DATA[4]
        .1
        .lines()
        .filter_map(|l| {
            let v: Vec<i64> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            (v.len() == 4).then(|| [v[0], v[1], v[2], v[3]])
        })
        .collect()
}

fn files() -> Vec<(String, String)> {
    DATA.iter().map(|(p, t)| (p.to_string(), t.to_string())).collect()
}

fn state(s: State) -> String {
    format!("1 4 r_eshape {} {} {} {}", s[0], s[1], s[2], s[3])
}

/// The first run's lines, from the state s: the three positions of the
/// window (s and the next two), then the state of the frame after them.
pub fn begin_tail(s: State) -> String {
    format!("w0 := {}\nw1 := u:s_tep w0\nw2 := u:s_tep w1\nw3 := u:s_tep w2\nr_avel 2 t_ake_2 w0 c_at w1 c_at w2\nr_avel w3\n", state(s))
}

/// A step's lines: the window of the three positions through the
/// network, its pixels, the likeliest place, and the true world moved on.
pub fn step_tail(window: &[Place; 3], truth: State) -> String {
    let xys: Vec<String> = window.iter().map(|(x, y)| format!("{x} {y}")).collect();
    format!(
        "W := 1 432 r_eshape r_avel u:f_rames 3 2 r_eshape {}\npr := u:p_redict W\nr_avel pr\nr_avel u:p_lace pr\nr_avel u:s_tep {}\n",
        xys.join(" "),
        state(truth)
    )
}

/// The first run, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Begin {
    pub window: [Place; 3],
    /// The state whose frame the first prediction is of.
    pub truth: State,
}

pub fn begin(s: State) -> Result<Begin, String> {
    let (out, _) = output_with(&files(), &format!("{}{}", head(), begin_tail(s)), 2)?;
    let w = numbers::<i64>(&out[0], 6)?;
    let t = numbers::<i64>(&out[1], 4)?;
    Ok(Begin { window: [(w[0], w[1]), (w[2], w[3]), (w[4], w[5])], truth: [t[0], t[1], t[2], t[3]] })
}

/// A step, as X_eTaL computed it.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    /// The predicted frame, a probability per pixel (SIDE x SIDE).
    pub probs: Vec<f64>,
    /// The likeliest place for the ball in it.
    pub place: Place,
    /// The true world one step on from the state the prediction was of.
    pub next: State,
}

pub fn step(window: &[Place; 3], truth: State) -> Result<Step, String> {
    let (out, _) = output_with(&files(), &format!("{}{}", head(), step_tail(window, truth)), 3)?;
    let place = numbers::<i64>(&out[1], 2)?;
    let n = numbers::<i64>(&out[2], 4)?;
    Ok(Step { probs: numbers(&out[0], SIDE * SIDE)?, place: (place[0], place[1]), next: [n[0], n[1], n[2], n[3]] })
}

/// Whether the predicted frame is the true one exactly: every pixel on
/// the right side of one half.
pub fn exact(probs: &[f64], truth: State) -> bool {
    let (x, y) = (truth[0] as usize, truth[1] as usize);
    probs.iter().enumerate().all(|(i, &p)| {
        let (r, c) = (i / SIDE, i % SIDE);
        (p > 0.5) == ((y..y + 2).contains(&r) && (x..x + 2).contains(&c))
    })
}
