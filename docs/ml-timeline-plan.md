# ML timeline: plan

The user, 2026-11-11: an "ML Timeline" like the Forth, Lisp and APL
timelines (web-sw-cor24-forth's, web-sw-cor24-macrolisp's
`lisp-timeline/`, sw-apl's), from about the 1950s: the milestones of
machine learning, its algorithms and key papers, and callouts with
pictures for this repository's demos (the CNN among them), each
picture a screenshot of the demo's live page, each callout a link to
it. Many more papers and demos fall in recent years than in the
early ones, so the strip is much wider and needs better navigation:
to the start and the end, and to a year; a minimap of the years over
a main view of months per year; sparse years kept short.

## What it is

- A page of its own, `pages/ml-timeline/` (live at
  `https://softwarewrighter.github.io/X_eTaL-ML/ml-timeline/`), linked
  from the catalog (pages/index.html) and the README. Not a demo: it
  runs no X_eTaL, so it is not in demos/ and has no CLI baseline.
- Its source in `timeline/`, its own Cargo workspace, forked from
  web-sw-cor24-macrolisp's `lisp-timeline/` (itself forked from
  sw-apl's by way of the Forth one; read only, copied and credited in
  its README): a core crate (events, scale, layout; std and serde
  only, natively tested) and a web crate (the strip, the minimap, the
  navigation), with `data/timeline.json` the events and
  `site/index.html` the shell.
- Built by `scripts/build-timeline.sh` (its tests, then wasm-pack into
  pages/ml-timeline/, the demos' pictures made into webp), called by
  `scripts/build-pages.sh`, so `just pages`, `just publish` and `just
  check-live` cover it; `just timeline` builds it alone.

## Events

Each event as the Lisp timeline's, `{year, when, title, text, href,
kind, ...}`, with a month:

- `year` and `month` (1-12, or absent when only the year is known:
  then the year's middle); `when` the words shown ("June 2017",
  "1958", "ca. 1966").
- `kind`: `paper` (a key paper: authors, title, venue, a link to it,
  arXiv or DOI where there is one), `milestone` (a system, a result or
  an event: the Perceptron Mark I, Deep Blue, AlphaGo, ImageNet),
  `note` (a line of context, no box of its own beyond its title), or
  `demo` (one of this repository's demos: its screenshot, its
  one-line summary from demo.toml, a link to its live page).
- `topics`: a few tags (`neural networks`, `CNN`, `attention`,
  `MoE`, `fine-tuning`, `clustering`, `embeddings`, `generative`,
  `reinforcement learning`, `optimization`, `quantization`) for a
  filter later, and to tie a demo to the papers it shows.
- `demo` events name their slug (`"demo": "cnn-digits"`); the build
  fills in the title, summary, picture and link from demos/<slug>/,
  so nothing is written twice. A demo stands at the year of the idea
  it shows (cnn-digits at LeCun et al. 1989, attention at 2017,
  lora-explorer at 2021), beside that paper's event, and says so.

Every date, author and title is checked against the paper or a
reliable source before it goes in; `docs/ml-timeline-sources.md`
lists each event's source. Pictures are this repository's own
screenshots only (no copied photographs or figures).

## The view: a minimap of years over months

- **The minimap**, a band across the top that always fits the window:
  every year from 1943 to 2026 at the same width, a tick for each
  event (a demo's tick marked as one), the decades labeled, and a
  frame showing what the main view shows. Click or drag in it to move
  the main view there.
- **The main view**, the strip of callouts, months within each year:
  an event stands at its month. A year's width is what its callouts
  need, laid out above and below the axis as the Lisp timeline lays
  them (the least-squares placement that keeps the years' order), at
  least a minimum (about 120 px for a year with an event), so
  2017-2026, with dozens of papers and demos, are wide and the 1950s
  narrow. The month ticks show only where a year is wide enough to
  label them.
- **Sparse years**: a year with no event takes a narrow fixed width; a
  run of three or more empty years folds into one break on the axis
  ("1974-1979"), so a quiet decade is not a blank screen. The minimap
  keeps every year, so the folds never hide where the view is.
- **Navigation**: `|<-` (to the start), `<<-` and `->>` (back and on
  by most of a window), `->|` (to the end, today and the newest
  demos); a year box (type 1998 and go, or pick a decade); the
  keyboard (Home, End, the arrow keys, Page Up and Down); the URL's
  hash (`#1989`, `#2017-06`) opens there and follows the view, so a
  year can be linked. The view opens at the start (or at the hash).
- At phone width the callouts stack narrower and the minimap stays
  one line; nothing scrolls the page sideways but the strip itself.

## Steps (Saga 6, after vera, before adapters-deploy)

| # | Step slug | Delivers |
| - | --------- | -------- |
| 10a | timeline-shell | `timeline/` forked from lisp-timeline: the core with months, kinds and topics; a scale whose years are as wide as their callouts, empty years folded; the minimap; the navigation (`\|<-`, `<<-`, `->>`, `->\|`, the year box, keys, the hash); native tests (scale, folds, layout, the hash's parsing); a few events to show it; `scripts/build-timeline.sh` and `just timeline`; built by `just pages`; a browser check |
| 10b | timeline-history | the events from 1943 to 2026, papers and milestones, each checked and listed in `docs/ml-timeline-sources.md`; a test that every event has a source and a well-formed date |
| 10c | timeline-demos | a `demo` event for every live demo, at the idea it shows, its webp made at build time from demos/<slug>/screenshot.png (`just screenshots` refreshed first), its summary and link from demo.toml; a test that every live demo has exactly one; the catalog, the README and each demo's README link the timeline; published and checked live |

## The events to check (a first list, not yet verified)

Papers and milestones the history step starts from, each still to be
checked; the demos each would stand beside are in brackets.

- 1943 McCulloch and Pitts, a logical calculus of nervous activity;
  1949 Hebb, The Organization of Behavior.
- 1950 Turing, Computing Machinery and Intelligence; 1952 Samuel's
  checkers program (1959: the term "machine learning"); 1956 the
  Dartmouth workshop; 1957 Lloyd's k-means (published 1982), 1967
  MacQueen [k-means]; 1958 Rosenblatt's perceptron [backprop's
  forerunner]; 1960 Widrow and Hoff, ADALINE and the LMS rule; 1967
  Cover and Hart, nearest neighbors; 1969 Minsky and Papert,
  Perceptrons.
- 1970 Linnainmaa, reverse-mode differentiation; 1974 Werbos; 1980
  Fukushima, the Neocognitron; 1982 Hopfield; 1986 Rumelhart, Hinton
  and Williams, backpropagation [backprop, net-macro, train-live];
  1989 LeCun et al., a CNN for zip codes [cnn-digits, cnn-backprop];
  1990 Church and Hanks, word association (PMI) [embedding-explorer];
  1991 Jacobs, Jordan, Nowlan and Hinton, adaptive mixtures of local
  experts [moe-router]; 1992 TD-Gammon; 1995 Cortes and Vapnik,
  support vector networks; 1997 Hochreiter and Schmidhuber, LSTM;
  Deep Blue; 1998 LeCun et al., LeNet-5 and MNIST.
- 2001 Breiman, random forests; 2006 Hinton et al., deep belief nets;
  2009 ImageNet; 2010 Nair and Hinton, ReLU [activations]; 2012
  AlexNet; 2013 Mikolov et al., word2vec [embedding-explorer];
  Kingma and Welling, VAE; 2014 Goodfellow et al., GANs; Bahdanau et
  al., attention; Kingma and Ba, Adam [train-live]; 2015 He et al.,
  ResNet; Sohl-Dickstein et al., diffusion; 2016 AlphaGo; Li et al.,
  ternary weight networks [ternary-net]; 2017 Vaswani et al., the
  Transformer [attention]; Shazeer et al., the sparsely-gated MoE
  [moe-router]; 2018 Radford et al., GPT [micro-gpt]; Devlin et al.,
  BERT [micro-bert]; Ha and Schmidhuber, World Models [world-model];
  2020 Ho et al., DDPM [diffusion]; Shazeer, GLU variants (SwiGLU);
  Brown et al., GPT-3; 2021 Hu et al., LoRA [lora-explorer]; Fedus et
  al., Switch Transformers; 2022 ChatGPT; 2023 Dettmers et al., QLoRA;
  Zhang et al., LoRA-FA; Mixtral (December 2023, the paper January
  2024) [moe-finetune]; 2024 Kopiczko et al., VeRA; Ma et al., BitNet
  b1.58 [ternary-net].
- 2026: X_eTaL and this repository's own demos that stand for no one
  paper [pipelines], at the end.
