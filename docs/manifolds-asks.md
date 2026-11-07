# Asks from X_eTaL-manifolds

What [X_eTaL-manifolds](https://github.com/softwarewrighter/X_eTaL-manifolds)
asks of this repository, filed in its `docs/xetal-ML-asks.md` (GM1,
GM2, ...), with this repository's word on each. That repository pins a
commit of this one (`XETAL_ML_COMMIT`), so an ask marked landed is
usable there once its pin passes the commit named.

Status: **landed** (in the commit named), **planned** (a step of
`docs/plan.md` named), **waiting** (on something named),
**declined** (with its reason).

Read on 2026-11-06 at their file's state of that day.

| # | Status | What landed or is planned | Where |
| - | ------ | ------------------------- | ----- |
| GM1 | landed (step manifolds-asks) | X_eTaL-fine-tuning's teaching shell is `shared/lesson` here, beside `shared/microscope`, its dependencies paths inside this repository. A dependent names at build time (`.cargo/config.toml` `[env]`): `LESSON_ROOT` (its root: its lessons' titles, its own libraries, its commit), `LESSON_SITE` (`name\|url\|catalog title`), `LESSON_GLOSSARY` (its glossary files, read after the shared base of the language's terms, a row replacing one of the same id), `LESSON_PINS` (the footer's pins, `Name=pin file`), `LESSON_LIBRARIES_PIN`; each library is linked to the repository it lives in at that repository's commit, and to its page in the dependent's cross-reference by its path under the root. `chrome::Frame::Other(name)` adds a kind beside the six. `lesson.css` is beside `microscope.css`. The tests run for this repository and, by hand, with X_eTaL-fine-tuning as the dependent; the switch for that repository is in `shared/lesson/README.md` (its GF1) | `shared/lesson` |
| GM2 | waiting | on your list (step foundation): which scripts you could call unchanged and what each of the others needed changed; taken from that list | |
| GM3 | landed (step manifolds-asks) | `microscope::turning`: a `Scene` of points (a color, a label), segments and vectors, and wire lines (a surface's meridians and parallels, computed by the page's program), `Angle` (yaw, pitch) as the page's state, `turn` (a drag), `key` (the arrow keys), `typical`, `project` (the projection, stated in the module), `marks` (far to near), and `Turning`, the SVG turned by pointer, touch and keys, a point picked by a click or a tap. The embedding explorer draws with its projection and turns by its keys; `stageview` keeps its own blocks for now | `shared/microscope/src/turning.rs` |
