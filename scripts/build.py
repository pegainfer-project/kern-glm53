#!/usr/bin/env python3
"""Build every module kernels.toml names into build/<name>.cubin and check its sha256.

A module is either compiled from `source` with its `flags` and `includes` sets and `defines`,
or copied from `prebuilt`. `--update` rewrites the pinned hashes instead of failing on a change.
Every build also lands in kern's blob cache under its hash, where the runtime finds it.

nvcc names a translation unit's internal symbols after the source's absolute path, so `source/` is
compiled from one fixed copy (STAGE, under a lock), its include directories with it; and cicc suffixes
some of those names with its own PID, which the PTX loses (`unpid`) before ptxas sees it. Every checkout,
worktree and container builds the same bytes.
"""
import argparse
import contextlib
import fcntl
import hashlib
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOML = ROOT / "kernels.toml"
STAGE = Path("/tmp/kern-glm53-build")


def build(name, k, cfg, nvcc):
    dst = ROOT / "build" / f"{name}.cubin"
    if "prebuilt" in k:
        shutil.copyfile(ROOT / k["prebuilt"], dst)
        return dst
    flags = cfg["flags"][k.get("flags", "plain")]
    incs = [f"-I{i}" for s in k.get("includes", []) for i in cfg["includes"][s]]
    defs = [f"-D{a}={b}" for a, b in sorted(k.get("defines", {}).items())]
    arch = f"-arch={cfg['toolchain']['arch']}"
    ptx = STAGE / f"{name}.ptx"
    subprocess.run([nvcc, "-ptx", arch, *flags, *incs, *defs, "-o", str(ptx), k["source"]], cwd=STAGE, check=True)
    ptx.write_bytes(unpid(ptx.read_bytes()))
    subprocess.run([nvcc, "-cubin", arch, "-o", str(dst), str(ptx)], cwd=STAGE, check=True)
    return dst


INTERNAL = rb"(?:_INTERNAL_|_GLOBAL__N__)[0-9a-f]{8}_\d+_\w+?_cu_[0-9a-f]{8}_"


def unpid(ptx):
    """A translation unit's internal names, `<len><prefix><pid>` with the PID cicc ran as, as `<len'><prefix>0`."""
    m = re.search(rb"_ZN(\d+)(" + INTERNAL + rb")(\d+)", ptx)
    if not m:
        return ptx
    pid = m[3][:int(m[1]) - len(m[2])]

    def fix(n):
        digits, prefix = n[1], n[2]
        width = str(len(prefix) + len(pid)).encode()
        assert digits.endswith(width), n[0]
        return digits[:-len(width)] + str(len(prefix) + 1).encode() + prefix + b"0"

    return re.sub(rb"(\d+)(" + INTERNAL + rb")" + re.escape(pid), fix, ptx)


def head(path):
    return subprocess.run(["git", "-C", str(ROOT / path), "describe", "--always", "--dirty"], capture_output=True,
                          text=True, check=True).stdout.strip()


def stage(path, stamp):
    dst = STAGE / path
    if (dst / ".stamp").is_file() and (dst / ".stamp").read_text() == stamp:
        return
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(ROOT / path, dst, ignore=shutil.ignore_patterns(".git"))
    (dst / ".stamp").write_text(stamp)


@contextlib.contextmanager
def staged(cfg):
    STAGE.mkdir(exist_ok=True)
    dirs = {i for v in cfg["includes"].values() for i in v}
    tops = [d for d in dirs if not any(d != o and d.startswith(o + "/") for o in dirs)]
    with open(STAGE.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for d in tops:
            stage(d, head(d))
        shutil.rmtree(STAGE / "source", ignore_errors=True)
        shutil.copytree(ROOT / "source", STAGE / "source")
        yield


def cache(sha, path):
    d = Path(os.environ.get("KERN_CACHE_DIR", Path.home() / ".cache" / "kern")) / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, d / sha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="pin the hashes of this build")
    ap.add_argument("names", nargs="*", help="modules to build (default: all)")
    a = ap.parse_args()
    cfg = tomllib.loads(TOML.read_text())
    nvcc = os.environ.get("NVCC", "nvcc")
    ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    want = cfg["toolchain"]["nvcc"]
    if f"V{want}" not in ver:
        raise SystemExit(f"nvcc {want} expected (kernels.toml), got: {ver.strip().splitlines()[-1]}")
    (ROOT / "build").mkdir(exist_ok=True)
    text, bad = TOML.read_text(), []
    with staged(cfg):
        built = {name: build(name, k, cfg, nvcc) for name, k in cfg["kernels"].items() if not a.names or name in a.names}
    for name, path in built.items():
        k = cfg["kernels"][name]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        cache(sha, path)
        ok = sha == k.get("sha256")
        print(f"{name}: {sha}{'' if ok else ' (differs from kernels.toml)'}", flush=True)
        if not ok and a.update:
            head = f'[kernels."{name}"]\n'
            i = text.index(head)
            j = text.find("\n[", i + len(head))
            j = len(text) if j < 0 else j + 1
            block = re.sub(r'sha256 = "[0-9a-f]*"\n', "", text[i:j])
            text = text[:i] + block.replace(head, f'{head}sha256 = "{sha}"\n') + text[j:]
        elif not ok:
            bad.append(name)
    if a.update:
        TOML.write_text(text)
    if bad:
        raise SystemExit("built cubins differ from kernels.toml: " + ", ".join(bad))


if __name__ == "__main__":
    main()
