# The ML timeline

A timeline of machine learning, from the McCulloch-Pitts neuron (1943)
to today: the key papers and milestones, and, beside the ideas they
show, this repository's demos. Live at
<https://softwarewrighter.github.io/X_eTaL-ML/ml-timeline/>.

## What it shows

- **The minimap**, across the top, always fits the window: every year
  from 1943 to 2026 at the same width, a tick for each event (a demo's
  in orange), the decades labeled, the folded runs of years shaded, and
  a frame around what the strip shows. Click it to center the strip
  there, or drag along it.
- **The strip**, below, is months within years. Each event stands at
  its month (or mid-year when only the year is known), its callout above
  or below the axis. A year is as wide as its callouts need, at least
  120 pixels, so the busy recent years are wide and the 1950s narrow; a
  year with no event is 40 pixels; a run of three or more empty years
  folds into one break on the axis ("1975-1979"). Months are ticked in a
  year wide enough, and lettered in a wider one.
- **Moving**: `|<-` to the start, `<<-` and `->>` back and on by most
  of a window, `->|` to the end; the year box takes `1989`, `2017-06` or
  a decade (`1990s`, also offered in its list); Home, End, Page Up, Page
  Down and the arrow keys; the wheel moves the strip sideways. The
  address's hash follows the view (`#1989`, `#2017-06`), so a year can
  be linked, and a page opened with a hash opens there.
- **Kinds**: a paper (its authors and venue, a link to it, by DOI or
  arXiv), a milestone (a system, a result, an event), a note (context),
  and a demo (its screenshot, linked to its live page, beside the idea
  it shows; the summary over the picture when the pointer is on it).
  Each event has a few topics.

## How it is built

- `data/timeline.json` -- the events, each `{year, month, when, title,
  text, href, kind, topics, demo, shows, image, tip, side}`; sorted by
  time when read. 84 papers, milestones and notes from 1943 to 2026,
  and a demo event for each of the 20 live demos. A demo event names
  only its slug, its place (the idea's month), what it shows and its
  topics; its title, its words (the summary's first clause), its tip
  (the whole summary) and its link come from `demos/<slug>/demo.toml`,
  and its picture is `shots/<slug>.webp`, so nothing is written twice.
- `../docs/ml-timeline-sources.md` -- where each event comes from: a
  row for every event but the demos, with its year, month and title as
  the event writes them, what was checked (authors, title, venue, date)
  and the link it was checked at, which must be the event's. How a
  date is chosen (an arXiv paper at its first version's month, a
  journal paper at its issue's) is written there.
- `crates/ml-timeline` -- free of the browser, tested natively: the
  events and their checks (a year on the axis, a month 1 to 12, a
  `when` naming its year and its month alone, a demo with its slug and
  what it shows, a paper with its link); the sources' table read and
  held to the events (`sources`, `unsourced`); the demos read from
  their `demo.toml` and filled in (`demo`, `fill`), every live demo on
  the timeline exactly once (`undemoed`); the scale (the segments: a
  year with events as wide as its callouts, an empty year narrow, a
  run folded; a year with its fraction to pixels and back); the
  layout; the moves, the hash and the minimap's mapping.
- `crates/ml-timeline-web` -- the page in Yew: the controls, the
  minimap, the axis, the callouts; `build.rs` gathers the demos'
  `demo.toml` files into the page; `index.html` is its shell and trunk
  entry. A picture is as tall as a short window allows, so the strip
  fits the window.
- `../scripts/build-timeline.sh` -- the tests, trunk, then each live
  demo's picture: `demos/<slug>/screenshot.png` (`just screenshots`
  refreshes them) made into a 600-pixel webp with `cwebp`.
- `browser.txt` -- what the built page must show in headless Chrome.

A new live demo needs its event here (the test fails without it), and
its README a link to its place (`scripts/demos.py check` fails
without it).

```bash
just timeline                          # its tests, then trunk, into pages/ml-timeline/
scripts/browser-check.sh ml-timeline   # the built page in headless Chrome
cd timeline && cargo test              # the core's tests alone
```

`just pages` builds it with the rest of the site, and the gate runs its
tests and its browser check whenever `timeline/`, or a demo's
`demo.toml` or screenshot, changes.

## Credits

Forked from the Lisp timeline of
[web-sw-cor24-macrolisp](https://github.com/sw-embed/web-sw-cor24-macrolisp)
(`lisp-timeline/`), itself from [sw-apl](https://github.com/sw-vibe-coding/sw-apl)'s
by way of the Forth one: the callouts' layout above and below the axis
(the least-squares placement that keeps the events' order), the leaders
and the strip are theirs. The months, the kinds and topics, the scale
of segments with its folds, the minimap and the navigation are new here.

## License

MIT, as the repository.
