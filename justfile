# X_eTaL-ML tasks. Recipes call scripts/*.sh, which hold the logic and
# work without just too. `just` alone lists the recipes.

set positional-arguments

# List the recipes
default:
    @just --list

# Snapshot a committed ref of ../X_eTaL into vendor/xetal/ (default HEAD); commit it on its own
vendor ref="HEAD":
    scripts/vendor-xetal.sh "$1"

# Snapshot libraries (default Check) from a committed ref of ../X_eTaL-libraries into vendor/xetal-libraries/; commit it on its own
vendor-libs ref="HEAD" *names:
    scripts/vendor-libraries.sh "$@"

# Build the vendored xetal CLI into target/xetal/
xetal:
    @scripts/build-xetal.sh

# The vendored X_eTaL: what was vendored (VENDORED) and the binary's version
xetal-version:
    @cat vendor/xetal/VENDORED
    @"$(scripts/build-xetal.sh)" --version | head -1

# Evaluate an expression with the vendored xetal, every library on XETAL_PATH: just eval "'+ r_/ 1 2 3"
eval expr:
    @scripts/xt eval -e "$1"

# Check the vendored X_eTaL: CLI builds, answers, names its commit; xetal-play usable natively and for wasm32
check-vendor:
    scripts/check-vendor.sh

# The demos, in catalog order
demos:
    @scripts/demos.py list

# Start a demo sub-project from demos/_template: just new-demo attention "Attention microscope"
new-demo slug title:
    scripts/new-demo.sh "$1" "$2"

# Run a demo's program (default SLUG.xtl) with the vendored xetal: just run moe-router
run slug file="":
    @scripts/run-demo.sh "$1" ${2:+"$2"}

# Run a demo's program as a notebook: each statement drawn, then its output
show slug file="":
    @scripts/run-demo.sh --echo "$1" ${2:+"$2"}

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

# Fetch MNIST into work/mnist/, train the cnn-digits demo's network and write its weights into cnn-digits.xtl (then just bless cnn-digits)
cnn-train:
    scripts/mnist.sh
    cargo run --release -q --manifest-path demos/cnn-digits/train/Cargo.toml

# Load one demo's built page in headless Chrome and check it shows X_eTaL's results
browser-check slug:
    scripts/browser-check.sh "$1"

# Build the live site into pages/ (committed; the Pages workflow publishes it)
pages:
    scripts/build-pages.sh

# Serve the built pages/ as GitHub Pages will: http://127.0.0.1:8435/X_eTaL-ML/ (8435 is this repo's port; each X_eTaL repo has its own)
serve-pages port="8435":
    scripts/serve-pages.sh "$1"

# Serve one demo's web app locally, rebuilt on change: just serve moe-router
serve slug port="8435":
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

# A library's exported names and their types: just types NN
types name:
    @scripts/xt type "libs/$1/src/$1.xtl"

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

# The full pre-commit gate
gate:
    scripts/gate.sh

# Show the agentrail saga state and the current step
status:
    agentrail status

# Open the saga plan
plan:
    @cat docs/plan.md
