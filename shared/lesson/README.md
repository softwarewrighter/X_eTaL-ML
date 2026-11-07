# lesson: the teaching shell

The parts a page needs to teach, on top of `microscope` (which runs X_eTaL,
reads arrays back and draws decorated source): frames and asides, a
glossary opened from the words themselves, library functions explained
from their own source, linked code, three depths, heat maps with a
legend, a stepper, written-out sums, a narrator, predictions. Written
in X_eTaL-fine-tuning for its lessons (d78aca4) and shared from here
at X_eTaL-manifolds' ask GM1 (X_eTaL-fine-tuning's GF1), so every
repository that pins X_eTaL-ML reaches it through that pin, built
against the one X_eTaL its pages and command line run. Presentation
only: nothing here computes a number a page shows.

| Module | What it gives a page |
| ------ | -------------------- |
| `chrome` | `SITE` and `header(title, lede)`: microscope's header, told whose page it is (`microscope::chrome::Site`, from `LESSON_SITE`); `footer()`: copyright, license, GitHub, the way back to the catalog, the cross-reference, each pin the dependent names (`LESSON_PINS`, linked to its commit), and the build, compact and without labels (host, short commit, `yyyymmddThhmm`); `pins()`, `pin(name)`, `ml_commit()` |
| `chrome` | `frame(kind, title, body)`: a framing section of a page, of one of six kinds (`Frame`: the question, the background, the rationale, the purpose, the summary, what is next), or a dependent's own (`Frame::Other("proof")`), each marked `data-frame` with its name for the dependent's browser check |
| `chrome` | `aside(id, summary, body)`: something one reader wants explained and another skips, hidden until its summary is opened; `panel(title, code, note, body)` |
| `eli5` | every idea explained simply, where the reader asks, at the depths they ask for: the glossary: this crate's `glossary.tsv` (the language's terms), then the dependent's files (`LESSON_GLOSSARY`), a row of the same id replacing one here (an id, the term, a few plain sentences, and for some a worked example and the lesson that teaches it); `Here`, a context naming the page, so an idea does not point to the lesson it is in; prose marked `[shown\|id]` and `**bold**`, drawn by `prose`, `p`, `li`: a marked word is a button that opens its explanation under it (keyboard and touch too); `index(texts)` lists every idea a page links, each closed until opened; `missing(texts)` finds marks naming no idea, for a page's test |
| `explain` | a library function from the pinned library's own source: `imports(program)`, `calls(program)`, `definition(library, name)` (its doc comment, its definition, its line, the link to that line on GitHub at the commit of the repository it lives in, its entry in the dependent's cross-reference), `function(alias, d, purpose, reading)` (the aside), `table(program)` (every function a program calls, in the library's words). A page never retypes library code |
| `code` | X_eTaL drawn as microscope draws it, with every function a link into the cross-reference: `Linker::of(program)` (its imports), `Linker::within(library, source)`, `.code(src)`, `.block(src)`; `builtins(src)` lists the built-in words a program uses, for a page to explain |
| `mode` | three depths over one computation: `Mode` (Beginner, Explorer, Advanced), `when(mode, depth, body)`, and `Switch`, radio buttons. The mode is the page's state, apart from the experiment's: switching it resets nothing |
| `heat` | X_eTaL-ML's `microscope::heat` colors with what a lesson needs: `Heat` (a legend, a note on whether the scale is shared, cells marked as read or written by the step on show, an inspector naming the matrix, row, column and value; optionally the number written in each cell, and a `pending` map drawn empty until its stage is reached; marked cells come in one after another); one tab stop, arrow keys over the cells, a click or tap to pick; `key(scale, shared)` for one legend under a group of maps |
| `stepper` | a pipeline stepped through from its recorded trace: `trace(line, statement)` reads what `@ flow:s_tages<` printed (each stage's value, shape, text and what it reads); `Stepper` (Start, Back, Play or Pause, Next, a bar showing how far along it is and, while playing, how long until the next stage); `flow(names, step, reads)` draws the stages in a row with a dot traveling to the stage on show from each stage it reads. Stepping moves a position in the record and runs nothing; motion is off under `prefers-reduced-motion` |
| `arithmetic` | a sum written out: `numeric` (each factor, each product, the total, all as X_eTaL printed them), `symbolic` (the same in symbols), `show`, `commas`, `list` for numbers; `adds_up` for a page's test |
| `narrate` | a change in words from measured numbers: `moved`, `count`, `held_out`. They compare and never do arithmetic or call a result better; `forbidden` finds a verdict word, for a test |
| `predict` | `Predict`: a question answered before the experiment under it is drawn; the right answer is given by the page from its run |

## A dependent says whose pages they are

At build time, in its `.cargo/config.toml` (`relative = true` makes a
path absolute from that file):

```
[env]
MICROSCOPE_LIBS = { value = "libs:work/xetal-libraries/libs", relative = true }
LESSON_ROOT     = { value = ".", relative = true }
LESSON_SITE     = "X_eTaL-fine-tuning|https://github.com/softwarewrighter/X_eTaL-fine-tuning|All lessons"
LESSON_GLOSSARY = "shared/glossary.tsv"
LESSON_PINS     = "X_eTaL-ML=XETAL_ML_COMMIT:X_eTaL-libraries=XETAL_LIBRARIES_COMMIT:X_eTaL=work/xetal-ml/XETAL_COMMIT"
```

| Name | What | Unset |
| ---- | ---- | ----- |
| `LESSON_ROOT` | the dependent's root: its `demos/*/demo.toml` (the lessons' titles), its `libs/` (its own libraries), its git commit (the footer's build) | this repository |
| `LESSON_SITE` | `name\|url\|catalog title`, the site of the header and footer | `X_eTaL-ML`, its GitHub URL, `All demos` |
| `LESSON_GLOSSARY` | glossary files under the root, `:`-separated, read after `glossary.tsv` | the base alone |
| `LESSON_PINS` | `Name=pin file` under the root, `:`-separated, the footer's pins in order | `X_eTaL-libraries` and `X_eTaL`, this repository's |
| `LESSON_LIBRARIES_PIN` | the pin file of the libraries named in `MICROSCOPE_LIBS` | `XETAL_LIBRARIES_COMMIT` |

Each library is linked to the repository it lives in at that
repository's commit: the dependent's own (`LESSON_ROOT/libs`, at its
HEAD), X_eTaL-ML's (this repository, at the clone's commit), and
X_eTaL-libraries' (`MICROSCOPE_LIBS`, at `LESSON_LIBRARIES_PIN`); a
later root wins a name both have, as microscope embeds them. Its page
in the dependent's cross-reference is its path under `LESSON_ROOT`,
slashes as dashes (`work/xetal-ml/libs/LoRA/src/LoRA.xtl` gives
`../doc/work-xetal-ml-libs-LoRA-src-LoRA.xtl.html`).

A page's `Cargo.toml` depends on both crates through the pinned clone:

```
lesson     = { path = "../../../work/xetal-ml/shared/lesson" }
microscope = { path = "../../../work/xetal-ml/shared/microscope" }
```

and its `index.html` links both stylesheets (trunk `rel="css"`):
`work/xetal-ml/shared/microscope/microscope.css` and
`work/xetal-ml/shared/lesson/lesson.css`.

## For X_eTaL-fine-tuning, switching to it

The modules and their functions are those of its `shared/lesson` at
d78aca4, with these differences:

- `chrome::pins()` gives every pin as `(name, full commit)`; `pin(name)`
  one short commit; `libraries_pin()` is `pin("X_eTaL-libraries")`.
- `chrome::Frame` has `Other(name)` beside the six.
- `REPO` and `SITE` come from `LESSON_SITE`; `ML` is unchanged.
- `glossary.tsv` here holds the language's thirteen terms (xetal,
  right-to-left, function, library, tuple, inner, arrow, note, yes-no,
  list, table, times, stage), without their lesson column; its own
  file, named in `LESSON_GLOSSARY`, keeps its other rows and may give
  these again with its lessons.
- `explain::ORIGINS` rows are `(file, repository, commit, path, page)`.

Its pages then drop its `shared/lesson` and set the environment above.

## Tests

`tests/lesson.rs` (`cargo test`, in the full gate): the shell built for
this repository; `a_dependent_names_its_site_glossary_and_pins` runs
when the crate is built with `LESSON_SITE` set (checked by hand with
X_eTaL-fine-tuning as the dependent, its glossary and pins).
