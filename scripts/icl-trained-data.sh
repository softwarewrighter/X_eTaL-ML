#!/usr/bin/env bash
# Train the icl-trained demo's three transformers and write their weights,
# logs and the analysis prompts into demos/icl-trained/data/. Each run is
# the one the demo ships: its seed, its steps and its thread count (the
# threads split the batch, and the order the gradient is summed in
# changes the last bits). Runs go in work/icl-runs/; about 5 hours on an
# M1 Max, the three in parallel. Other seeds are in the demo's README.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
cargo build --release -q --manifest-path demos/icl-trained/train/Cargo.toml
bin="$root/target/release/icl-trained-train"
runs="$root/work/icl-runs"
data="$root/demos/icl-trained/data"
mkdir -p "$runs"
train() { "$bin" --cond "$1" --seed "$2" --steps "$3" --threads "$4" --d 64 --ffn 128 --every "$5" --out "$runs/$1-$2" > "$runs/$1-$2.log" 2>&1; }
train mix 0 25000 2 500 &
train recurring 2 100000 8 1000 &
train gate 0 50000 3 500 &
wait
"$bin" --d 64 --ffn 128 --export "$runs/mix-0" "$data/mix"
"$bin" --d 64 --ffn 128 --export "$runs/recurring-2" "$data/recurring"
"$bin" --d 64 --ffn 128 --export "$runs/gate-0" "$data/gate"
"$bin" --prompts "$data"
echo "icl-trained: data/ written (then just bless icl-trained)"
