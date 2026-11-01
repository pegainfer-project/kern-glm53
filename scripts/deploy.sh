#!/usr/bin/env bash
# scripts/deploy.sh <kern checkout> <out> [gen.py flags, e.g. --kv bf16]: GLM-5.3's packed prefill (16 sequences,
# 16384 rows a call, 1M context, fp8 latent cache) cut into PP4 stages at layers 18, 38 and 58, each verified
# by the checkout's kern.
#   <out>/glm53.json         the whole model
#   <out>/p/stage{0..3}.json the stages
# Needs build/glm_flashmla.cubin (scripts/build.py) for the attention's entry name.
set -euo pipefail
kern=$(realpath "$1") out=$(realpath -m "$2")
shift 2
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=$kern/target/release/kern
export KERN=$kern KERN_INDEX_DIR=${KERN_INDEX_DIR:-$kern/kernel_index}
mkdir -p "$out"
python3 "$repo/scripts/gen.py" "$@" > "$out/glm53.json"
"$bin" verify "$out/glm53.json"
at=$(python3 -c "
import json, sys
calls = json.load(open(sys.argv[1]))['programs']['prefill']['calls']
print(','.join(str(next(i for i, c in enumerate(calls) if c['label'] == f'L{l}.plan')) for l in (18, 38, 58)))
" "$out/glm53.json")
rm -rf "$out/p"
"$bin" cut "$out/glm53.json" --at "$at" --out "$out/p"
