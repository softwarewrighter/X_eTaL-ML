#!/usr/bin/env python3
"""A demo as a paced notebook, for reading at a terminal or recording.

Runs `scripts/run-demo.sh --echo SLUG` (each statement drawn, then its
output) and passes the output on:
  - every line clipped to the terminal's width (ANSI colours kept),
    an ellipsis where it was cut;
  - a run of data statements (longer than the width and mostly number
    literals, as weights and sample digits are) shown as its first
    line and one "... N more" line;
  - a pause (PACE seconds, default 0.35) before each statement, so the
    program unfolds rather than arriving at once.

  scripts/tour.py SLUG [WIDTH] [PACE]
"""
import os
import re
import shutil
import subprocess
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def visible(s):
    return len(ANSI.sub("", s))


def clip(line, width):
    if visible(line) <= width:
        return line
    out, n, i = [], 0, 0
    while i < len(line) and n < width - 1:
        m = ANSI.match(line, i)
        if m:
            out.append(m.group())
            i = m.end()
        else:
            out.append(line[i])
            n += 1
            i += 1
    return "".join(out) + "\x1b[0m…"


def data(line, width):
    text = ANSI.sub("", line)
    if len(text) <= width or "\u2190" not in text:
        return False
    rhs = text.split("\u2190", 1)[1]
    numeric = sum(c in "0123456789.- " for c in rhs)
    return numeric > 0.8 * len(rhs)


def main():
    slug = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else shutil.get_terminal_size().columns
    pace = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    run = subprocess.run([os.path.join(ROOT, "scripts", "run-demo.sh"), "--echo", slug],
                         capture_output=True, text=True)
    lines = run.stdout.splitlines()
    run_ = 0  # consecutive data statements so far

    def more():
        if run_ > 1:
            print(f"      \x1b[2m\u235d ... {run_ - 1} more line(s) like it (data)\x1b[0m", flush=True)

    for line in lines:
        statement = line.startswith("      ")
        if statement and data(line, width):
            run_ += 1
            if run_ > 1:
                continue
        else:
            more()
            run_ = 0
        if statement:
            time.sleep(pace)
        print(clip(line, width), flush=True)
    more()
    sys.stderr.write(run.stderr)
    sys.exit(run.returncode)


if __name__ == "__main__":
    main()
