"""What a change affects, for the gate (scripts/gate.sh).

Reads the files changed since HEAD (tracked, staged and untracked
alike; GATE_BASE names another base, such as origin/main) and prints
one line per thing the gate should test:

  demo SLUG    a demo whose own files changed, or that imports a changed
               library (u_se< "Name"), or every demo when the shared page
               shell, the tooling or a pin changed: its CLI baselines,
               cargo test, the wasm32 check and its page in the browser
  lib NAME     a library whose own files changed, or that imports a
               changed one: its baselines, pinned types and expansions,
               its doc examples
  pages        the libraries' site (site/) and the pages of the demos
               above are built; the libs site runs in the browser
  all          everything (the full gate): the shared shell, scripts,
               the justfile, a pin, the templates or this planner changed

The cheap checks (the tooling self-tests, spelling, asks, rebind,
literal c_at chains, markdown) always run; a change to the docs alone
plans nothing else.

  scripts/py scripts/affected.py              # the plan for the present change
  scripts/py scripts/affected.py --self-test  # the planner on known changes
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A change here affects everything.
EVERYTHING = ("shared/", "scripts/", "justfile", "XETAL_COMMIT", "XETAL_LIBRARIES_COMMIT", "templates/", ".python-version")
# Documentation and records: the cheap checks only.
DOCS_ONLY = (".md", ".agentrail/", ".gitignore", "LICENSE", ".github/")


def demos():
    return sorted(p.name for p in (ROOT / "demos").iterdir() if (p / "demo.toml").exists() and p.name != "_template")


def libs():
    return sorted(p.name for p in (ROOT / "libs").iterdir() if (p / "src").is_dir())


def imports(path):
    """The libraries a directory's X_eTaL (and Rust, for a page's programs) names with u_se<."""
    names = set()
    for p in list(path.rglob("*.xtl")) + list(path.rglob("*.xtlm")) + list(path.rglob("*.rs")):
        if "target" in p.parts:
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        names.update(re.findall(r'u_se<\s*\\?"([A-Z][A-Za-z0-9]*)\\?"', text))
    return names


def changed(base=None):
    base = base or os.environ.get("GATE_BASE", "HEAD")
    out = subprocess.run(["git", "diff", "--name-only", base], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files = set(out.split())
    out = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    files.update(out.split())
    return sorted(files)


def plan(files, demo_names, lib_names, demo_imports, lib_imports):
    """The plan's lines for the changed files: see the module's doc."""
    files = [f for f in files if not (f.endswith(DOCS_ONLY[0]) or f.startswith(DOCS_ONLY[1:]))]
    if not files:
        return []
    if any(f.startswith(EVERYTHING) or f == "justfile" for f in files):
        return ["all"]
    touched_demos = {f.split("/")[1] for f in files if f.startswith("demos/") and f.count("/") >= 2}
    touched_libs = {f.split("/")[1] for f in files if f.startswith("libs/") and f.count("/") >= 2}
    site = any(f.startswith("site/") for f in files)
    if any(not (f.startswith(("demos/", "libs/", "site/", "bench/"))) for f in files):
        return ["all"]  # a file the planner does not know
    # A changed library affects what imports it, transitively.
    affected_libs = set(touched_libs)
    while True:
        more = {l for l in lib_names if lib_imports[l] & affected_libs} - affected_libs
        if not more:
            break
        affected_libs |= more
    affected_demos = set(touched_demos) | {d for d in demo_names if demo_imports[d] & affected_libs}
    lines = [f"demo {d}" for d in sorted(affected_demos)] + [f"lib {l}" for l in sorted(affected_libs)]
    if affected_demos or affected_libs or site:
        lines.append("pages")
    return lines


def present():
    demo_names, lib_names = demos(), libs()
    demo_imports = {d: imports(ROOT / "demos" / d) for d in demo_names}
    lib_imports = {l: imports(ROOT / "libs" / l) for l in lib_names}
    return plan(changed(), demo_names, lib_names, demo_imports, lib_imports)


def self_test():
    demo_names = ["cnn-digits", "net-macro", "diffusion"]
    lib_names = ["NN", "Net", "Conv", "Flow"]
    demo_imports = {"cnn-digits": {"NN", "Conv"}, "net-macro": {"NN", "Net"}, "diffusion": {"NN", "Conv"}}
    lib_imports = {"NN": set(), "Net": {"NN"}, "Conv": set(), "Flow": {"Strings"}}
    cases = [
        ([], []),
        (["README.md", "docs/plan.md", "CHANGES.md"], []),
        (["demos/diffusion/diffusion.xtl"], ["demo diffusion", "pages"]),
        (["libs/Flow/src/Flow.xtlm", "libs/Flow/docs/README.md"], ["lib Flow", "pages"]),
        (["libs/Conv/src/Conv.xtl"], ["demo cnn-digits", "demo diffusion", "lib Conv", "pages"]),
        (["libs/NN/src/NN.xtl"], ["demo cnn-digits", "demo diffusion", "demo net-macro", "lib NN", "lib Net", "pages"]),
        (["site/src/lib.rs"], ["pages"]),
        (["shared/microscope/src/run.rs"], ["all"]),
        (["XETAL_COMMIT"], ["all"]),
        (["scripts/gate.sh"], ["all"]),
        (["bench/dense.xtl"], []),  # the timings are not in the gate
    ]
    for files, want in cases:
        got = plan(files, demo_names, lib_names, demo_imports, lib_imports)
        if got != want:
            sys.exit(f"affected: self-test failed for {files}: planned {got}, expected {want}")
    print("affected: self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif sys.argv[1:]:
        sys.exit("usage: scripts/py scripts/affected.py [--self-test]")
    else:
        print("\n".join(present()))
