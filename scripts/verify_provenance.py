# -*- coding: utf-8 -*-
"""Check which commit of this repository holds the code that produced each result file.

Result files record their code in one or more of these forms:
  * {"rcdp_guard_sha256": {name: sha256}, "scripts_sha256": {name: sha256}}           (end-to-end harness)
  * {"code": {"sha256": {name: sha256}}} or any other {name.py: sha256} map            (TAP, policy cost)
  * {"sha256_rcdp_guard_and_runner" | "sha256": H, "files": [names]}, where H is the SHA-256 over name + bytes of the
    sorted rcdp_guard/*.py followed by the runner / extra scripts                        (single-turn, multi-turn, cost)
For each result file the script lists the commits (by tag, in history order) whose files reproduce every recorded
hash. Files that record no code are reported as such.

usage: python scripts/verify_provenance.py <results dir, e.g. the unpacked rcdp_r1_results_redacted/results> \
           [--repo .] [--json out.json]
"""
import argparse
import hashlib
import itertools
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

HEX = re.compile(r"^[0-9a-f]{64}$")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True).stdout


def states(repo):
    order = git(repo, "rev-list", "--reverse", "HEAD").decode().split()
    tags = defaultdict(list)
    for line in git(repo, "show-ref", "--tags", "-d").decode().splitlines():
        sha, ref = line.split()
        name = ref.split("refs/tags/", 1)[1]
        tags[sha].append(name[:-3] if name.endswith("^{}") else name)
    out = []
    for c in order:
        name = sorted(set(tags.get(c, [])))
        if not name and c != order[0]:
            continue
        files = {}
        for line in git(repo, "ls-tree", "-r", c).decode().splitlines():
            meta, path = line.split("\t", 1)
            if path.endswith((".py", ".sh")):
                files[path] = hashlib.sha256(git(repo, "cat-file", "blob", meta.split()[2])).hexdigest()
        out.append({"tag": ",".join(name) or "submitted", "commit": c, "files": files})
    return out


def extract(o):
    files, combos = set(), set()

    def walk(x, key=""):
        if isinstance(x, dict):
            if isinstance(x.get("files"), list) and all(isinstance(f, str) for f in x["files"]):
                for hk in ("sha256_rcdp_guard_and_runner", "sha256"):
                    if isinstance(x.get(hk), str) and HEX.match(x[hk]):
                        combos.add((x[hk], tuple(x["files"])))
            scope = "rcdp_guard/" if key == "rcdp_guard_sha256" else "scripts/" if key == "scripts_sha256" else ""
            for k, v in x.items():
                if isinstance(v, str) and HEX.match(v) and k.endswith((".py", ".sh")):
                    files.add((scope, k, v))
                elif isinstance(v, (dict, list)):
                    walk(v, k)
        elif isinstance(x, list):
            for v in x[:20]:
                walk(v, key)

    walk(o)
    return files, combos


def file_ok(state, scope, name, h):
    base = Path(name).name
    return any(p.startswith(scope) and Path(p).name == base and s == h for p, s in state["files"].items())


def combo_ok(repo, state, h, names, cache={}):
    guard = sorted(p for p in state["files"] if p.startswith("rcdp_guard/") and p.count("/") == 1 and p.endswith(".py"))
    k = len(guard)
    if [Path(p).name for p in guard] != list(names[:k]):
        return False
    extras = []
    for n in names[k:]:
        c = [p for p in state["files"] if Path(p).name == n and p.startswith("scripts/")]
        if not c:
            return False
        extras.append(c)

    def blob(p):
        key = (state["commit"], p)
        if key not in cache:
            cache[key] = git(repo, "show", f"{state['commit']}:{p}")
        return cache[key]

    for combo in itertools.product(*extras):
        hh = hashlib.sha256()
        for p in guard + list(combo):
            hh.update(Path(p).name.encode())
            hh.update(blob(p))
        if hh.hexdigest() == h:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    st = states(a.repo)
    rows = []
    for f in sorted(a.results.rglob("*.json")):
        try:
            o = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        files, combos = extract(o)
        if not files and not combos:
            rows.append({"file": f.relative_to(a.results).as_posix(), "states": None})
            continue
        ok = [s["tag"] for s in st if all(file_ok(s, *r) for r in files) and all(combo_ok(a.repo, s, *c) for c in combos)]
        rows.append({"file": f.relative_to(a.results).as_posix(), "states": ok})
    n_rec = sum(r["states"] is not None for r in rows)
    n_ok = sum(bool(r["states"]) for r in rows)
    print(f"{len(rows)} result files; {n_rec} record their code; {n_ok} of these match a commit of this repository")
    for r in rows:
        if r["states"] is not None:
            print(f"  {r['file']}: {r['states'][0] if r['states'] else 'NO MATCH'}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
