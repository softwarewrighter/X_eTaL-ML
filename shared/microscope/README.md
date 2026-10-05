# The microscope: the demo pages' shared shell

Every demo page runs an X_eTaL program in the browser, reads back the
arrays it prints, and draws them beside the program's decorated
source. This crate (`microscope`) is what the pages share.

| Module | What it gives a page |
| ------ | -------------------- |
| `run` | `output(src, lines)` runs a program (the pinned `xetal-play`) and returns its printed lines, or the X_eTaL error; `numbers(line, n)` reads one printed array (`r_avel`); `lit(x)` writes a Float as an X_eTaL literal (plain decimal: X_eTaL literals have no exponent); `matrix(name, rows, cols, items)` binds a matrix; `section(src, start, end)` cuts the core out of a demo's `.xtl`; `now()` times a run |
| `source` | `code(src)` draws any snippet decorated and colored, as X_eTaL renders it; `line(src, range)` and `block(src, range)` draw a line or a program with `range` highlighted as one block; `between` and `find` compute ranges; `shape(dims, meaning)` labels a shape as `s_hape = ...` with a tooltip |
| `canvas`, `color` | `Canvas` draws an RGBA array scaled to its box and reports clicks as (row, column); `color` turns arrays into pixels (`field`, `scaled`, `signed`, `mask`, `ramp`) |
| `cells` | small boards as clickable HTML cells (0 / 1 boards, shaded counts with numbers) |
| `chrome` | `header` (logo, title, lede), `chip` (a stage: name, code, shape), `panel`, `notice`, `footer` (copyright, license, repository, all demos, the pinned X_eTaL commit, build host, sha and time) |

`microscope.css` is the shared stylesheet (light and dark themes, the
highlight, panels, chips, canvases); a page links it with
`<link data-trunk rel="css" href="../../../shared/microscope/microscope.css" />`.

## A new demo's web app

1. `just new-demo SLUG "Title"`, then write `SLUG.xtl`: the program,
   with its core between two marker comments (for example
   `# -- the grid ...` and `# -- end of the core`), and its golden.
2. `demos/SLUG/web/`, a Cargo workspace like
   `demos/mandelbrot/web/`: `Cargo.toml` depending on
   `microscope = { path = "../../../shared/microscope" }`, and an
   `index.html` with the trunk links (the rust binary, the logo, the
   favicon, `microscope.css`).
3. `src/micro.rs`, the model, tested natively in `tests/`:
   `include_str!("../../SLUG.xtl")`, the core cut out with `section`,
   a program that sets the page's parameters (with `lit`), runs the
   core and prints each array to show with `r_avel`; parse with
   `output` and `numbers`. Test that the page runs the CLI file's core,
   and test the arrays against what the arithmetic must give.
4. `src/view.rs`: the stages and the range of the core computing each.
5. `src/model.rs` (state and actions, a Yew reducer) and `src/app.rs`
   (the page from `chrome`, `source`, `canvas` or `cells`).
6. `web/browser.txt`: lines of text the page shows only once X_eTaL
   has run (an inspector heading, a computed number), for the browser
   test.
7. `just serve SLUG` while working; `just pages`; `just bless SLUG`
   creates the reg-rs baselines (`reg/cli-*` for each `.xtl`,
   `reg/browser-SLUG` for the built page in headless Chrome); review
   them, then `just test-demo SLUG`; set `status = "live"` in
   `demo.toml`.

Conventions: every code snippet on a page is drawn decorated, never as
typed ASCII; shapes are shown as `s_hape`; an X_eTaL error keeps the
last good state and shows a notice; anything X_eTaL lacks goes in
`docs/xetal-asks.md` and the demo's README.
