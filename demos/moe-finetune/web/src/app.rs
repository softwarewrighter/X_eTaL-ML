//! The page: the tokens to fine-tune on and how many seeds; every
//! method's measures gathered as X_eTaL finishes each run (one at a
//! time, the page drawn between); for the method picked, where the
//! tokens go before and after, how large each update is, the loss; the
//! four adapters compared at equal ranks and at equal budgets, at a
//! rate chosen; the program.

use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use gloo_timers::callback::Timeout;
use yew::prelude::*;

use microscope::chrome::{footer, header, notice, panel};
use microscope::run::now;
use microscope::source::{block, NONE};

use crate::micro::{compare, compare_tail, head, run, tail, Adapter, Compared, Config, Data, Method, Rate, Run, STEPS};

/// A method's run: on the tokens from a seed.
pub type MethodRun = (Method, Data, usize);
/// A comparison's run: an adapter at a configuration's rank, at a rate,
/// on the tokens from a seed.
pub type CompareRun = (Adapter, Config, Rate, Data, usize);

/// A run to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Job {
    Method(MethodRun),
    Compare(CompareRun),
}

impl Job {
    fn title(self) -> String {
        match self {
            Job::Method((m, _, seed)) => format!("{}, seed {seed}", m.title()),
            Job::Compare((a, c, _, _, seed)) => format!("{} at rank {}, seed {seed}", a.title(), c.rank(a)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Model {
    pub data: Data,
    pub seeds: usize,
    /// Fine-tuning asked for: until then only the base runs.
    pub started: bool,
    pub picked: Method,
    pub results: BTreeMap<MethodRun, Rc<Run>>,
    /// The comparison asked for, its rate, its runs.
    pub comparing: bool,
    pub rate: Rate,
    pub compared: BTreeMap<CompareRun, Rc<Compared>>,
    pub queue: VecDeque<Job>,
    /// Milliseconds of the last run.
    pub ms: f64,
    pub notice: Option<String>,
}

pub enum Action {
    Data(Data),
    Seeds(usize),
    Pick(Method),
    /// Fine-tune every method.
    Start,
    /// Compare the four adapters.
    Compare,
    Rate(Rate),
    /// Run the next job in the queue.
    Next,
}

impl Model {
    fn new() -> Self {
        let mut m = Model { data: Data::Mixed, seeds: 1, started: false, picked: Method::None, results: BTreeMap::new(), comparing: false, rate: Rate::Methods, compared: BTreeMap::new(), queue: VecDeque::new(), ms: 0.0, notice: None };
        m.plan();
        m
    }

    /// Queue every run the settings show and no run has given yet: each
    /// method in order, a seed at a time (the base alone until started);
    /// then, once asked, each adapter at each configuration.
    fn plan(&mut self) {
        let methods: &[Method] = if self.started { &Method::ALL } else { &[Method::None] };
        for seed in 1..=self.seeds {
            for &m in methods {
                let job = (m, self.data, seed);
                if !self.results.contains_key(&job) && !self.queue.contains(&Job::Method(job)) {
                    self.queue.push_back(Job::Method(job));
                }
            }
        }
        if self.comparing {
            for seed in 1..=self.seeds {
                for c in Config::ALL {
                    for a in Adapter::ALL {
                        let job = (a, c, self.rate, self.data, seed);
                        if !self.compared.contains_key(&job) && !self.queue.contains(&Job::Compare(job)) {
                            self.queue.push_back(Job::Compare(job));
                        }
                    }
                }
            }
        }
    }

    /// The comparison's runs of an adapter at a configuration, by seed.
    pub fn compares(&self, a: Adapter, c: Config) -> Vec<Rc<Compared>> {
        (1..=self.seeds).filter_map(|s| self.compared.get(&(a, c, self.rate, self.data, s)).cloned()).collect()
    }

    /// The runs of a method at the settings shown, by seed.
    pub fn runs(&self, m: Method) -> Vec<Rc<Run>> {
        (1..=self.seeds).filter_map(|s| self.results.get(&(m, self.data, s)).cloned()).collect()
    }
}

impl Reducible for Model {
    type Action = Action;
    fn reduce(self: Rc<Self>, action: Action) -> Rc<Self> {
        let mut m = (*self).clone();
        match action {
            Action::Data(d) => {
                m.data = d;
                m.plan();
            }
            Action::Seeds(n) => {
                m.seeds = n;
                m.plan();
            }
            Action::Pick(p) => m.picked = p,
            Action::Compare => {
                m.comparing = true;
                m.plan();
            }
            Action::Rate(r) => {
                m.rate = r;
                m.plan();
            }
            Action::Start => {
                m.started = true;
                m.picked = Method::Added;
                m.plan();
            }
            Action::Next => {
                if let Some(job) = m.queue.pop_front() {
                    let t = now();
                    let done = match job {
                        Job::Method(k) => run(k.0, k.1, k.2, STEPS).map(|r| {
                            m.results.insert(k, Rc::new(r));
                        }),
                        Job::Compare(k) => compare(k.0, k.1, k.2, k.3, k.4, STEPS).map(|r| {
                            m.compared.insert(k, Rc::new(r));
                        }),
                    };
                    m.notice = done.err().map(|e| format!("{}: {e}", job.title()));
                    m.ms = now() - t;
                }
            }
        }
        Rc::new(m)
    }
}

fn act(m: &UseReducerHandle<Model>, a: impl Fn() -> Action + 'static) -> Callback<MouseEvent> {
    let d = m.dispatcher();
    Callback::from(move |_| d.dispatch(a()))
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// A mean with its range over the seeds, when more than one.
fn spread(v: &[f64]) -> String {
    let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    match v.len() {
        0 => "..".into(),
        1 => format!("{:.3}", v[0]),
        _ => format!("{:.3} ({lo:.3} to {hi:.3})", mean(v)),
    }
}

fn controls(m: &UseReducerHandle<Model>) -> Html {
    let status = match (m.queue.front(), m.started) {
        (Some(job), _) => format!("X_eTaL is fine-tuning: {} ({} runs to go; the page answers between runs)", job.title(), m.queue.len()),
        (None, true) => "every run done".to_string(),
        (None, false) => "the base, measured".to_string(),
    };
    let start = (!m.started).then(|| html! { <button class="on" onclick={act(m, || Action::Start)}>{"Fine-tune every method"}</button> });
    let data = |d: Data, label: &str| html! { <button class={classes!((m.data == d).then_some("on"))} onclick={act(m, move || Action::Data(d))}>{label.to_string()}</button> };
    let seeds = |n: usize, label: &str| html! { <button class={classes!((m.seeds == n).then_some("on"))} onclick={act(m, move || Action::Seeds(n))}>{label.to_string()}</button> };
    html! { <>
        <div class="controls">
            { start }
            <span>{"Fine-tune on: "}</span>
            { data(Data::New, "the 200 new tokens alone") }
            { data(Data::Mixed, "mixed with 200 old ones (replay)") }
            <span>{" Seeds: "}</span>
            { seeds(1, "1") }
            { seeds(3, "1, 2 and 3") }
        </div>
        <div class="controls"><span class="gen">{format!("{status} \u{00b7} {STEPS} steps of Adam a run \u{00b7} the last run took {:.1} s", m.ms / 1000.0)}</span></div>
    </> }
}

/// Every method's measures at the settings shown.
fn table(m: &UseReducerHandle<Model>) -> Html {
    let base = m.runs(Method::None).first().map(|r| r.old);
    let rows = Method::ALL.iter().map(|&meth| {
        let runs = m.runs(meth);
        let new: Vec<f64> = runs.iter().map(|r| r.new).collect();
        let old: Vec<f64> = runs.iter().map(|r| r.old).collect();
        let change = match (base, old.is_empty()) {
            (Some(b), false) => {
                let c = 100.0 * (mean(&old) - b);
                let ok = c >= -3.0;
                html! { <td class={classes!("num", if ok { "pass" } else { "fail" })}>{format!("{c:+.1}")}{ if ok { " \u{2713}" } else { "" } }</td> }
            }
            _ => html! { <td class="num">{".."}</td> },
        };
        let (n, b) = runs.first().map_or((String::from(".."), String::from("..")), |r| (r.numbers.to_string(), r.bytes.to_string()));
        let pick = act(m, move || Action::Pick(meth));
        html! {
            <tr class={classes!("method", (m.picked == meth).then_some("picked"))} onclick={pick}>
                <td><b>{meth.title()}</b><br /><span class="what">{meth.trains()}</span></td>
                <td class="num">{n}<br /><span class="what">{format!("{b} bytes")}</span></td>
                <td class="num">{spread(&new)}</td>
                <td class="num">{spread(&old)}</td>
                { change }
                <td class="num">{format!("{} of {}", runs.len(), m.seeds)}</td>
            </tr>
        }
    });
    html! {
        <table class="facts methods">
            <tr><th>{"Method"}</th><th>{"Trains"}</th><th>{"New task"}</th><th>{"Old task"}</th><th>{"Old, in points"}</th><th>{"Seeds"}</th></tr>
            { for rows }
        </table>
    }
}

/// The four adapters at each configuration: what each trains, the new
/// task's gain over the base and the old task's change, in points, and
/// the gain per number trained.
fn comparison(m: &UseReducerHandle<Model>) -> Html {
    let base = m.runs(Method::None).first().map(|r| (r.new, r.old));
    let rate = |r: Rate, label: &str| html! { <button class={classes!((m.rate == r).then_some("on"))} onclick={act(m, move || Action::Rate(r))}>{label.to_string()}</button> };
    let ask = (!m.comparing).then(|| html! { <button class="on" onclick={act(m, || Action::Compare)}>{"Compare the four adapters"}</button> });
    let tables = Config::ALL.iter().map(|&c| {
        let rows = Adapter::ALL.iter().map(|&a| {
            let runs = m.compares(a, c);
            let new: Vec<f64> = runs.iter().map(|r| r.new).collect();
            let old: Vec<f64> = runs.iter().map(|r| r.old).collect();
            let n = runs.first().map(|r| r.numbers);
            let (gain, change, per) = match (base, runs.is_empty(), n) {
                (Some((bn, bo)), false, Some(n)) => {
                    let g = 100.0 * (mean(&new) - bn);
                    (format!("{g:+.1}"), format!("{:+.1}", 100.0 * (mean(&old) - bo)), format!("{:+.2}", g / n as f64))
                }
                _ => ("..".into(), "..".into(), "..".into()),
            };
            html! { <tr>
                <td><b>{a.title()}</b><br /><span class="what">{a.trains()}</span></td>
                <td class="num">{c.rank(a)}</td>
                <td class="num">{n.map_or("..".to_string(), |n| n.to_string())}</td>
                <td class="num">{spread(&new)}</td>
                <td class="num">{gain}</td>
                <td class="num">{change}</td>
                <td class="num">{per}</td>
                <td class="num">{format!("{} of {}", runs.len(), m.seeds)}</td>
            </tr> }
        });
        html! { <>
            <p class="calc">{c.title()}</p>
            <table class="facts methods">
                <tr><th>{"Adapter"}</th><th>{"Rank"}</th><th>{"Trains"}</th><th>{"New task"}</th><th>{"Gain, points"}</th><th>{"Old, in points"}</th><th>{"Gain a number"}</th><th>{"Seeds"}</th></tr>
                { for rows }
            </table>
        </> }
    });
    html! { <>
        <div class="controls">{ ask }<span>{"Adam's rate: "}</span>{ rate(Rate::Methods, "0.02, the methods'") }{ rate(Rate::Fast, "0.1") }</div>
        { for tables }
    </> }
}

/// Before and after, a bar each, for each of the 5 experts.
fn routing(before: &[f64], after: &[f64]) -> Html {
    let rows = before.iter().zip(after).enumerate().map(|(e, (&b, &a))| {
        let name = if e == 4 { "5 (new)".to_string() } else { (e + 1).to_string() };
        html! {
            <tr>
                <td>{name}</td>
                <td><div class="hbar before" style={format!("width:{:.0}%", 100.0 * b)}></div><span class="v">{format!("{b:.3}")}</span></td>
                <td><div class="hbar after" style={format!("width:{:.0}%", 100.0 * a)}></div><span class="v">{format!("{a:.3}")}</span></td>
            </tr>
        }
    });
    html! { <table class="facts routing"><tr><th>{"Expert"}</th><th>{"Before"}</th><th>{"After"}</th></tr>{ for rows }</table> }
}

fn sizes(s: &[f64]) -> Html {
    let top = s.iter().cloned().fold(1e-9, f64::max);
    let names = ["expert 1", "expert 2", "expert 3", "expert 4", "shared layer", "new expert"];
    let rows = s.iter().zip(names).map(|(&v, name)| html! {
        <tr><td>{name}</td><td><div class="hbar after" style={format!("width:{:.0}%", 100.0 * v / top)}></div><span class="v">{format!("{v:.2}")}</span></td></tr>
    });
    html! { <table class="facts routing">{ for rows }</table> }
}

/// The loss after each step, on a linear scale.
fn losses(v: &[f64]) -> Html {
    if v.len() < 2 {
        return html! { <p class="calc">{"Nothing trained."}</p> };
    }
    let (w, h, ml) = (520.0, 160.0, 40.0);
    let hi = v.iter().cloned().fold(0.0, f64::max).max(1e-9);
    let px = |i: usize| ml + (w - ml - 10.0) * i as f64 / (v.len() - 1) as f64;
    let py = |y: f64| 8.0 + (h - 30.0) * (1.0 - y / hi);
    let pts: Vec<String> = v.iter().enumerate().map(|(i, &y)| format!("{:.1},{:.1}", px(i), py(y))).collect();
    html! {
        <figure class="chart">
            <svg width={w.to_string()} height={h.to_string()} viewBox={format!("0 0 {w} {h}")} role="img" aria-label="the loss after each step">
                <line x1={ml.to_string()} x2={(w - 10.0).to_string()} y1={py(0.0).to_string()} y2={py(0.0).to_string()} class="grid" />
                <text x={(ml - 6.0).to_string()} y={(py(hi) + 4.0).to_string()} class="tick end">{format!("{hi:.2}")}</text>
                <text x={(ml - 6.0).to_string()} y={(py(0.0) + 4.0).to_string()} class="tick end">{"0"}</text>
                <text x={px(0).to_string()} y={(h - 6.0).to_string()} class="tick mid">{"1"}</text>
                <text x={px(v.len() - 1).to_string()} y={(h - 6.0).to_string()} class="tick mid">{v.len().to_string()}</text>
                <polyline points={pts.join(" ")} class="line" />
            </svg>
            <figcaption>{format!("from {:.3} to {:.3}", v[0], v[v.len() - 1])}</figcaption>
        </figure>
    }
}

#[function_component(App)]
pub fn app() -> Html {
    let model = use_reducer(Model::new);
    {
        // The next run once this frame has drawn.
        let d = model.dispatcher();
        let key = (model.queue.len(), model.queue.front().cloned());
        use_effect_with(key, move |(n, _)| {
            let t = (*n > 0).then(|| Timeout::new(30, move || d.dispatch(Action::Next)));
            move || drop(t)
        });
    }
    let picked = model.picked;
    let detail = model.results.get(&(picked, model.data, 1)).cloned();
    let data_note = match model.data {
        Data::New => "the 200 new tokens alone",
        Data::Mixed => "the 200 new tokens mixed with 200 of the old task's",
    };
    let body = html! {
        <div class="layout even">
            <div class="col">
                { panel("Each method:", "(method, tokens, 120) u:t_une seed", "Every method fine-tunes the same base on the same tokens: kind 3's new rule. The new task is 200 held-out tokens of kind 3 by its new rule; the old task is the 300 held-out tokens of kinds 1, 2 and 4, whose rule did not change. The old task's change is in points against the base (3 points is the bar, ✓ within it). The numbers trained are adapters, a router or a new expert; the base's own weights stay frozen, and the bytes are those numbers in FP32. Click a method for its routing, its updates and its loss.", true, table(&model)) }
                { panel("Four adapters compared:", "(kind, r, tokens, 120, lr) u:c_ompare seed", "The research's comparison, on the same base and tokens, the router frozen: shared LoRA (on the shared layer, its attention LoRA), expert LoRA, LoRA-FA (each expert's A frozen) and shared VeRA (one pair of frozen random projections for every expert, each expert its own two vectors), at equal ranks and at about equal numbers trained. The gain is the new task's accuracy over the base's, in points; the gain a number is that over the numbers trained, the research's efficiency, never a substitute for the gain itself; the old task's change is in points too. An adapter with frozen projections moves slowly at the methods' rate (VeRA's paper trains at several times LoRA's), so the rate is yours to choose. Each run is about 15 seconds.", false, comparison(&model)) }
                { panel("Where the tokens go:", "m u:s_hare xn", &format!("{}, seed 1: the share of the new task's tokens, then of the old task's, each expert gets, before and after (each column sums to 2, the top 2). With the router frozen and the shared layer as it was, nothing can move; the shared layer's adapter moves the router's input, so it moves the routing too; a trained router or a new expert's own row moves it on purpose.", picked.title()), false, match &detail {
                    Some(r) => html! { <>
                        <p class="calc">{"The new task's tokens:"}</p>
                        { routing(&r.base_new, &r.share_new) }
                        <p class="calc">{"The old task's tokens:"}</p>
                        { routing(&r.base_old, &r.share_old) }
                    </> },
                    None => html! { <p class="calc">{"Waiting for its run."}</p> },
                }) }
                { panel("How large each update is:", "u:s_ize (A, B)", "The size of each adapter's update, |s B A| (the square root of the sum of its squares), and of the new expert's weights. With the router frozen, an expert the new tokens seldom reach learns little.", false, match &detail {
                    Some(r) => sizes(&r.sizes),
                    None => html! {},
                }) }
                { panel("The loss:", "losses", &format!("The cross-entropy on {data_note} after each of the {STEPS} steps."), false, match &detail {
                    Some(r) => losses(&r.losses),
                    None => html! {},
                }) }
            </div>
            <div class="col">
                <section class="panel code">
                    <h2>{"The program"}</h2>
                    <p class="note">{"Everything the page runs, in your browser, once for each method, tokens and seed: moe-finetune.xtl up to its end of the core (the tokens, the base read from its data, the model with its adapters, a method's step and its measures, the four adapters compared), then the run's own lines, here the picked method's."}</p>
                    { block(&format!("{}{}", head(), tail(picked, model.data, 1, STEPS)), NONE) }
                    <p class="note">{"And a comparison's run, here shared VeRA at equal budgets (rank 9) at the rate chosen, its own lines after the same core:"}</p>
                    { block(&compare_tail(Adapter::Vera, Config::EqualBudgets, model.rate, model.data, 1, STEPS), NONE) }
                </section>
            </div>
        </div>
    };
    html! {
        <>
        <header>
            { header("MoE fine-tuning", "A tiny mixture-of-experts (a shared layer, 4 experts, each token through its top 2) learned a task in X_eTaL; now one kind of token gets a new rule. Each way of fine-tuning it trains a few numbers beside the frozen base, on the same tokens: is the new task learned, and is the old one kept? Every run is X_eTaL training in your browser, one method after another (about 15 seconds each).") }
            { controls(&model) }
            { notice(&model.notice) }
        </header>
        <main>{body}</main>
        { footer() }
        </>
    }
}
