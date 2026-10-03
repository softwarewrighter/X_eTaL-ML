#!/usr/bin/env python3
"""Every example on a library's page is recorded output.

In libs/<Name>/docs/README.md, a code block holding session lines (an
expression indented six spaces, its result under it) must show, for
each expression, a result that appears verbatim in one of the
library's reg-rs baselines (tests/*.out, tests/*.err). So a page
cannot show a result its tests do not produce.
  scripts/check-examples.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bad = 0
for d in sorted((ROOT / "libs").glob("*/")):
    page = d / "docs" / "README.md"
    if not page.exists():
        continue
    recorded = "\n".join(p.read_text() for p in sorted((d / "tests").glob("*.out")) + sorted((d / "tests").glob("*.err")))
    for block in re.findall(r"```\n(.*?)```", page.read_text(), re.S):
        expr, result = None, []
        for line in block.split("\n") + ["      "]:
            if line.startswith("      "):
                if expr and result and "\n".join(result) not in recorded:
                    bad += 1
                    print(f"check-examples: {d.name}: {expr.strip()!r} shows {result[0]!r}..., not in its baselines", file=sys.stderr)
                expr, result = line, []
            elif line:
                result.append(line)
print(f"check-examples: {'ok' if bad == 0 else f'{bad} example(s) not recorded'}")
sys.exit(1 if bad else 0)
