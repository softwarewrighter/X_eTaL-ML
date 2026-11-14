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
  folds into one break on the axis ("1970-1985"). Months are ticked in a
  year wide enough, and lettered in a wider one.
- **Moving**: `|<-` to the start, `<<-` and `->>` back and on by most
  of a window, `->|` to the end; the year box takes `1989`, `2017-06` or
  a decade (`1990s`, also offered in its list); Home, End, Page Up, Page
  Down and the arrow keys; the wheel moves the strip sideways. The
  address's hash follows the view (`#1989`, `#2017-06`), so a year can
  be linked, and a page opened with a hash opens there.
- **Kinds**: a paper (its authors and venue, a link to it, by DOI or
  arXiv), a milestone (a system, a result, an event), a note (context),
  and a demo (its screenshot, linked to its live page). Each event has a
  few topics.

## How it is built

- `data/timeline.json` -- the events, each `{year, month, when, title,
  text, href, kind, topics, demo, image, tip, side}`; sorted by time
  when read. `docs/ml-timeline-sources.md` lists each event's source.
- `crates/ml-timeline` -- free of the browser, tested natively: the
  events and their checks (a year on the axis, a month 1 to 12, a `when`
  naming its year, a demo with its slug, a paper with its link); the
  scale (the segments: a year with events as wide as its callouts, an
  empty year narrow, a run folded; a year with its fraction to pixels
  and back); the layout; the moves, the hash and the minimap's mapping.
- `crates/ml-timeline-web` -- the page in Yew: the controls, the
  minimap, the axis, the callouts; `index.html` is its shell and trunk
  entry.
- `browser.txt` -- what the built page must show in headless Chrome.

```bash
just timeline                          # its tests, then trunk, into pages/ml-timeline/
scripts/browser-check.sh ml-timeline   # the built page in headless Chrome
cd timeline && cargo test              # the core's tests alone
```

`just pages` builds it with the rest of the site, and the gate runs its
tests and its browser check whenever `timeline/` changes.

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
