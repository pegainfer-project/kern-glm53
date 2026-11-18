# Sourced by every loop/ script: machine paths from loop/local.env, the repo root.
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$LOOP_DIR/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "${GLM_LOOP_ENV:-$LOOP_DIR/local.env}"
set +a
export LD_LIBRARY_PATH="$GLM_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export KERN=$GLM_KERN KERN_INDEX_DIR=$GLM_KERN/kernel_index
export TMPDIR="/dev/shm/glm53loop-$(id -un)"
mkdir -p "$TMPDIR"
