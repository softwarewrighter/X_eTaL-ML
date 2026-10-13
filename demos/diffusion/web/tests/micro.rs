use diffusion_web::micro::{head, on, run, seeded, LEVELS, N, PICTURES, SOURCE};
use diffusion_web::view::program;

fn file(text: &str) -> Vec<f64> {
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

#[test]
fn the_sampler_is_the_trainers_rust_one() {
    // The trainer's starting noise, eight times over, becomes its expected picture eight times.
    let noise = file(include_str!("../../data/noise.txt"));
    let want = file(include_str!("../../data/expected.txt"));
    assert_eq!((noise.len(), want.len()), (N, N));
    let j = run(&noise.repeat(PICTURES)).unwrap();
    assert_eq!(j.levels.len(), LEVELS + 1);
    for i in 0..PICTURES {
        let off = j.picture(LEVELS, i).iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(off < 1e-6, "picture {i}: {off}");
    }
    // The journey starts at the noise itself.
    assert_eq!(j.picture(0, 0), &noise[..]);
}

#[test]
fn the_same_noise_gives_the_same_pictures_and_most_are_shapes() {
    let noise = seeded(2026);
    let (a, b) = (run(&noise).unwrap(), run(&noise).unwrap());
    assert_eq!(a, b, "no chance inside: DDIM adds no noise");
    let shapes = (0..PICTURES).filter(|&i| (8..=40).contains(&on(a.picture(LEVELS, i)))).count();
    assert!(shapes >= 5, "{shapes} of {PICTURES} pictures have a shape's worth of pixels on");
    assert_ne!(seeded(1), seeded(2));
}

#[test]
fn the_page_runs_the_programs_core() {
    assert!(SOURCE.contains(head()));
    let src = program();
    assert!(src.contains("E := 8 100 r_eshape ...") && src.contains("r_avel L"));
    assert!(microscope::source::rebound(&src).is_empty(), "{:?}", microscope::source::rebound(&src));
}
