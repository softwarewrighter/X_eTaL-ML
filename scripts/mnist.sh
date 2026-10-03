#!/usr/bin/env bash
# Fetch MNIST (LeCun, Cortes and Burges) for the cnn-digits trainer into
# work/mnist/ (gitignored; only the trained weights are committed), from
# the PyTorch project's public mirror, check the files' MD5 sums, and
# unpack them (the trainer reads the raw IDX files).
#   scripts/mnist.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$root/work/mnist"; mkdir -p "$dir"
base="https://ossci-datasets.s3.amazonaws.com/mnist"
while read -r sum name; do
  f="$dir/$name"
  [ -s "$f" ] || curl -fsSL -o "$f" "$base/$name"
  got="$(md5 -q "$f" 2>/dev/null || md5sum "$f" | cut -d' ' -f1)"
  [ "$got" = "$sum" ] || { echo "mnist: $name has MD5 $got, expected $sum" >&2; rm -f "$f"; exit 1; }
  [ -s "${f%.gz}" ] || gunzip -kf "$f"
  echo "ok: $name"
done <<'SUMS'
f68b3c2dcbeaaa9fbdd348bbdeb94873 train-images-idx3-ubyte.gz
d53e105ee54ea40749a09fcbcd1e9432 train-labels-idx1-ubyte.gz
9fb629c4189551a2d022fa330f9573f3 t10k-images-idx3-ubyte.gz
ec29112dd5afa0611ce80d1b7f02629c t10k-labels-idx1-ubyte.gz
SUMS
