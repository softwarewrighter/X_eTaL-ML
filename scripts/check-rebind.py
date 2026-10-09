#!/usr/bin/env python3
"""Check that every name is bound once in its scope (X_eTaL lang-choices
M1, changed 2026-10-08): binding a name again in the same scope -- a
file's top level, or a lambda's body together with its parameters -- is
an error once X_eTaL implements it (its step 091). Shadowing an outer
name in a lambda stays legal; `_` binds nothing; a name ending in ! is a
variable and may be bound again.

Scans the tracked programs and libraries (*.xtl, *.xtlm) and the
libraries' pinned expansions (libs/*/tests/expand-*.out: the code the
macros write). The pages' generated programs are checked by their own
tests (microscope::source::rebound, the same rule).

  scripts/check-rebind.py            # the repository
  scripts/check-rebind.py FILE...    # these files
  scripts/check-rebind.py --self-test
"""
import re
import subprocess
import sys

NAME = r"[A-Za-z][A-Za-z0-9_:]*[?!]?"


def names(pattern):
    """The names a binding's left side or a lambda's parameters bind."""
    return [n for n in re.findall(NAME, pattern) if n != "_"]


def code_of(line):
    """A line without its comment (a # outside strings)."""
    out, quote = [], False
    for ch in line:
        if ch == '"':
            quote = not quote
        if ch == "#" and not quote:
            break
        out.append(ch)
    return "".join(out)


def rebound(text):
    """(line, name, first line) for each name bound again in its scope."""
    scopes = [{}]
    found = []
    for i, line in enumerate(text.split("\n"), 1):
        code = code_of(line)
        s = code.strip()
        m = re.match(r"^(" + NAME + r")\s*:=", s) or re.match(r"^\(([^()]*)\)\s*:=", s)
        if m:
            for n in names(m.group(1)):
                if n.endswith("!"):
                    continue
                if n in scopes[-1]:
                    found.append((i, n, scopes[-1][n]))
                else:
                    scopes[-1][n] = i
        opened = code.count("{") - code.count("}")
        head = re.search(r"\{([^{}]*)->\s*$", code)
        if opened > 0 and head:
            scopes.append({n: i for n in names(head.group(1)) if not n.endswith("!")})
        elif opened > 0:
            scopes.append({})
        while opened < 0 and len(scopes) > 1:
            scopes.pop()
            opened += 1
    return found


def self_test():
    bad = {
        "top": "a := 1\na := 2\n",
        "param": "u:f_ := { x ->\n  x := x + 1\n  x\n}\n",
        "pattern": "u:f_ := { (a, b) ->\n  b := a\n  b\n}\n",
        "local": "u:f_ := { x ->\n  h := x\n  h := h\n  h\n}\n",
        "destructure": "w := 1\n(w, v) := (2, 3)\n",
    }
    good = {
        "shadow": "x := 1\nu:f_ := { x ->\n  y := x\n  y\n}\n",
        "wildcard": "(_, a) := (1, 2)\n(_, b) := (3, 4)\n",
        "variable": "a! := 1\na! := 2\n",
        "separate lambdas": "u:f_ := { x ->\n  y := x\n  y\n}\nu:g_ := { x ->\n  y := x\n  y\n}\n",
        "comment": "a := 1 # a := 2\n# a := 3\n",
    }
    for k, src in bad.items():
        assert rebound(src), f"self-test: {k} not caught"
    for k, src in good.items():
        assert not rebound(src), f"self-test: {k} wrongly caught: {rebound(src)}"
    print("check-rebind: self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    files = argv[1:] or subprocess.run(
        ["git", "ls-files", "*.xtl", "*.xtlm", "libs/*/tests/expand-*.out"],
        capture_output=True, text=True, check=True).stdout.split()
    bad = 0
    for f in files:
        if "/_template/" in f or f.startswith("templates/"):
            continue
        for i, n, first in rebound(open(f).read()):
            print(f"{f}:{i}: {n} is already bound (line {first}): each name is bound once in its scope")
            bad += 1
    if bad:
        return 1
    print(f"check-rebind: ok ({len(files)} files, every name bound once in its scope)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
