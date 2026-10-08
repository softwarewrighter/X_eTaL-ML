# X_eTaL-ML tasks. Recipes call scripts/*.sh, which hold the logic and
# work without just too. `just` alone lists the recipes.

set positional-arguments

# List the recipes
default:
    @just --list

# Get and build xetal at the known-good commit in XETAL_COMMIT (clone in work/xetal, binary bin/xetal), and the libraries used from X_eTaL-libraries
xetal:
    @scripts/xetal.sh

# Pin a newer X_eTaL: a committed ref of ../X_eTaL (default HEAD) into XETAL_COMMIT, built; then test, bench-check and commit
xetal-pin ref="HEAD":
    scripts/xetal-pin.sh "$1"

# Pin a newer X_eTaL-libraries (for Check): a committed ref of ../X_eTaL-libraries into XETAL_LIBRARIES_COMMIT
libs-pin ref="HEAD":
    scripts/xetal-libraries.sh --pin "$1"

# The pinned X_eTaL: the commits in XETAL_COMMIT and XETAL_LIBRARIES_COMMIT, and the binary's own report
xetal-version:
    @echo "XETAL_COMMIT $(cat XETAL_COMMIT)"
    @echo "XETAL_LIBRARIES_COMMIT $(cat XETAL_LIBRARIES_COMMIT)"
    @"$(scripts/xetal.sh)" --version

# Evaluate an expression with the pinned xetal, every library on XETAL_PATH: just eval "'+ r_/ 1 2 3"
eval expr:
    @scripts/xt eval -e "$1"

# Check the pinned X_eTaL: CLI builds, answers, reports its commit; Check loads; xetal-play usable natively and for wasm32
check-xetal:
    scripts/check-xetal.sh

# The demos, in catalog order
demos:
    @scripts/demos.py list

# Start a demo sub-project from demos/_template: just new-demo attention "Attention microscope"
new-demo slug title:
    scripts/new-demo.sh "$1" "$2"

# Run a demo's program (default SLUG.xtl) with the pinned xetal: just run moe-router
run slug file="":
    @scripts/run-demo.sh "$1" ${2:+"$2"}

# Run a demo's program as a notebook: each statement drawn, then its output
show slug file="":
    @scripts/run-demo.sh --echo "$1" ${2:+"$2"}

# A demo's program after macro expansion (what its macro calls became): just expand net-macro
expand slug file="":
    @cd demos/$1 && XETAL_PATH="$(cd ../.. && ls -d libs/*/src work/libs/*/src 2>/dev/null | sed 's#^#../../#' | paste -sd: -)" "$(../../scripts/xetal.sh)" expand "${2:-$1.xtl}"

# A demo as a paced notebook: each statement, then its result; long lines clipped, data runs collapsed
tour slug:
    @scripts/tour.py "$1"

# Record demos at the command line (VHS tapes, demos/<slug>/<slug>.tape) as animated WebP: just record moe-router
record *slugs:
    scripts/record.sh "$@"

# Test one demo: its reg-rs baselines (CLI, page), web/ tests, test.sh
test-demo slug:
    scripts/test-demos.sh "$1"

# Accept one demo's current output as its reg-rs baselines, creating missing ones (review the diff!)
bless slug:
    XETAL_BLESS=1 scripts/test-demos.sh "$1"

# Train the ternary-net demo's network offline and write its weights into ternary-net.xtl (then just bless ternary-net)
ternary-train:
    cargo run --release -q --manifest-path demos/ternary-net/train/Cargo.toml

# Train the net-macro demo's networks offline (their specs read from net-macro.xtl) and write data/ (then just bless net-macro)
net-train:
    cargo run --release -q --manifest-path demos/net-macro/train/Cargo.toml

# Fetch MNIST into work/mnist/, train the cnn-digits demo's network and write its weights into cnn-digits.xtl (then just bless cnn-digits)
cnn-train:
    scripts/mnist.sh
    cargo run --release -q --manifest-path demos/cnn-digits/train/Cargo.toml

# Load one demo's built page in headless Chrome and check it shows X_eTaL's results
browser-check slug:
    scripts/browser-check.sh "$1"

# Build the live site into pages/ (not tracked; just publish publishes it)
pages:
    scripts/build-pages.sh

# The cross-reference (xetal doc) of every library and demo program, into pages/doc; just pages builds it too
doc:
    scripts/doc-site.sh

# Run every ## >> example in the libraries' doc comments (xetal doc --test); the gate runs it
doc-test:
    scripts/doc-test.sh

# Publish pages/ as the gh-pages branch's only commit (the live site); needs a clean work tree
publish:
    scripts/publish-pages.sh

# Verify the deployed site after a publish: the catalog reports this commit and every live page runs X_eTaL (headless Chrome)
check-live:
    scripts/check-live.sh

# Serve the built pages/ as GitHub Pages will: http://127.0.0.1:8435/X_eTaL-ML/ (8435 is this repo's port; each X_eTaL repo has its own)
serve-pages port="8435":
    scripts/serve-pages.sh "$1"

# Serve one demo's web app locally, rebuilt on change: just serve moe-router
serve slug port="8435": xetal
    cd demos/{{slug}}/web && trunk serve --release --port {{port}} --address 127.0.0.1

# Screenshot every demo (from the built pages/) into demos/<slug>/screenshot.png
screenshots *slugs:
    scripts/screenshots.sh "$@"

# The libraries: name, recommended alias, what it is
libs:
    @scripts/libs.py table | column -t -s "$(printf '\t')"

# The directories to put on XETAL_PATH: export XETAL_PATH="$(just path)"
path:
    @scripts/libs.py path

# Start a library from templates/Library: just new-lib NN nn: "neural network layers"
new-lib name alias summary:
    scripts/new-lib.sh "$1" "$2" "$3"

# Run a library's test programs (or one): just run-lib NN softmax
run-lib name prog="":
    @scripts/run-lib.sh "$1" ${2:+"$2"}

# Run a library's demos (or one): just demo-lib NN xor
demo-lib name prog="":
    @scripts/run-lib.sh --demos "$1" ${2:+"$2"}

# A library's demo as a notebook, each statement then its output
show-lib name prog="":
    @scripts/run-lib.sh --echo --demos "$1" ${2:+"$2"}

# What a macro library's demo becomes after macro expansion: just expand-lib Net xor
expand-lib name prog:
    @cd libs/$1/demos && ../../../scripts/xt expand "${2%.xtl}.xtl"

# A library's exported names and their types: just types NN
types name:
    @for f in libs/$1/src/$1.xtl libs/$1/src/$1.xtlm; do [ -f "$f" ] && scripts/xt type "$f"; done; true

# Test one library with reg-rs: pinned types, test programs, demos
test-lib name:
    scripts/test-libs.sh "$1"

# Create missing baselines and accept new output for one library (review the diff!)
bless-lib name:
    XETAL_BLESS=1 scripts/test-libs.sh "$1"

# Test everything: every demo and every library (XETAL_BROWSER=0 skips the browser checks)
test:
    scripts/test-demos.sh
    scripts/test-libs.sh

# Time the ML workloads (bench/*.xtl and every demo), best of RUNS, as a table for docs/speed.md
bench RUNS="3":
    scripts/bench.sh {{RUNS}}

# Check them against this machine's baseline: fails when one is more than 15% slower (shows the faster ones too)
bench-check RUNS="5":
    scripts/bench-check.sh {{RUNS}}

# Record this machine's baseline (blessing a slowdown needs the user's approval)
bench-bless RUNS="5":
    scripts/bench-check.sh --bless {{RUNS}}

# The full pre-commit gate
gate:
    scripts/gate.sh

# Show the agentrail saga state and the current step
status:
    agentrail status

# Open the saga plan
plan:
    @cat docs/plan.md
