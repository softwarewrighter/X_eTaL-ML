#!/usr/bin/env -S uv run --quiet --no-project --managed-python --python 3.12 python
"""Check docs/xetal-asks.md against X_eTaL's own ledger of asks (its
docs/asks.toml, read from the pinned copy in work/xetal): every ask in
our table must be in the ledger for X_eTaL-ML with the same state, or
be marked "not yet taken"; and the ledger must have no ask of ours that
our table lacks. So a decision X_eTaL makes (an ask landed, declined)
is noticed here at the next pin, and an ask filed here is not left
untaken.

  scripts/py scripts/check-asks.py
  scripts/py scripts/check-asks.py --self-test
"""
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATES = {"landed", "partly landed", "planned", "declined", "new", "not yet taken"}


def ours(text):
    """(ask, state) for each row of our table."""
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"^\| (M\d+) \| ([^|]+)\|", text, re.M)}


def theirs(toml_text):
    """(ask, state) for each of X_eTaL-ML's asks in the ledger."""
    return {a["id"]: a["state"] for a in tomllib.loads(toml_text).get("ask", []) if a.get("repo") == "X_eTaL-ML"}


def problems(mine, ledger):
    out = []
    for ask, state in sorted(mine.items(), key=lambda kv: int(kv[0][1:])):
        if state not in STATES:
            out.append(f"{ask}: status '{state}' is not one of {sorted(STATES)}")
        elif state == "not yet taken":
            if ask in ledger:
                out.append(f"{ask}: marked not yet taken, but X_eTaL's ledger has it as '{ledger[ask]}'")
        elif ask not in ledger:
            out.append(f"{ask}: '{state}' here, but not in X_eTaL's ledger (mark it 'not yet taken' until it is)")
        elif ledger[ask] != state:
            out.append(f"{ask}: '{state}' here, '{ledger[ask]}' in X_eTaL's ledger")
    for ask in sorted(set(ledger) - set(mine)):
        out.append(f"{ask}: in X_eTaL's ledger ('{ledger[ask]}') but not in docs/xetal-asks.md")
    return out


def self_test():
    toml = '[[ask]]\nid = "M1"\nrepo = "X_eTaL-ML"\nstate = "landed"\n[[ask]]\nid = "M2"\nrepo = "X_eTaL-ML"\nstate = "declined"\n[[ask]]\nid = "D1"\nrepo = "X_eTaL-demos"\nstate = "new"\n'
    led = theirs(toml)
    assert led == {"M1": "landed", "M2": "declined"}, led
    assert not problems({"M1": "landed", "M2": "declined", "M3": "not yet taken"}, led)
    assert problems({"M1": "landed", "M2": "open"}, led), "a stale state"
    assert problems({"M1": "landed"}, led), "a ledger ask missing here"
    assert problems({"M1": "landed", "M2": "declined", "M3": "new"}, led), "new but not in the ledger"
    assert problems({"M1": "not yet taken", "M2": "declined"}, led), "not yet taken but in the ledger"
    assert ours("| M7 | landed | bug | x |\n| M8 | not yet taken | speed | y |\n") == {"M7": "landed", "M8": "not yet taken"}
    print("check-asks: self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    ledger_file = ROOT / "work" / "xetal" / "docs" / "asks.toml"
    if not ledger_file.exists():
        print(f"check-asks: no {ledger_file} (just xetal)", file=sys.stderr)
        return 1
    found = problems(ours((ROOT / "docs" / "xetal-asks.md").read_text()), theirs(ledger_file.read_text()))
    for p in found:
        print(f"check-asks: {p}")
    if found:
        return 1
    print("check-asks: ok (every ask in step with X_eTaL's ledger at the pin, or not yet taken)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
