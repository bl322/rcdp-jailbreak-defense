"""Convert an external multi-turn dataset into the runner's dialogue JSONL (ledger R3-4: independent data).

The source schema is not assumed: name the fields explicitly. Each input record
(JSONL, or a JSON list) holds one conversation, either as a list of messages
{<role-field>: ..., <text-field>: ...} or as a list of strings.

    python scripts/multiturn_import_dialogues.py --input raw/<file>.jsonl \
        --output data/external_mt/<name>.jsonl --benchmark <Name> --label harmful \
        --messages-field messages --role-field role --text-field content --id-field id

Output records: example_id "<benchmark>-<id>", benchmark, label, turns (user /
assistant; system messages dropped; trailing assistant turns dropped because the
controller routes user turns), metadata.cluster_id (--cluster-field, else the
example id: each conversation is its own cluster) and metadata.source.
Use --label-field/--harmful-values when one file mixes labels.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_records(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("data", [])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def get(record, dotted):
    cur = record
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def convert_turns(items, args):
    turns = []
    for k, item in enumerate(items or []):
        if isinstance(item, str):
            role = ("user" if k % 2 == 0 else "assistant") if args.strings == "alternate" else "user"
            text = item
        else:
            raw_role = str(item.get(args.role_field, "")).strip().lower()
            text = item.get(args.text_field)
            if raw_role in args.system_roles:
                continue
            role = "user" if raw_role in args.user_roles else "assistant"
        if text is None or not str(text).strip():
            continue
        turns.append({"role": role, "text": str(text).strip()})
    while turns and turns[-1]["role"] != "user":
        turns.pop()
    if args.max_user_turns:
        out, n = [], 0
        for t in turns:
            n += t["role"] == "user"
            if n > args.max_user_turns:
                break
            out.append(t)
        turns = out
    return turns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--benchmark", required=True, help="name stored in each record (use the agnostic setting for it)")
    p.add_argument("--label", choices=["harmful", "benign"], help="label of every conversation")
    p.add_argument("--label-field", help="per-record label field (dotted path) instead of --label")
    p.add_argument("--harmful-values", default="harmful,unsafe,1,true", help="values of --label-field meaning harmful")
    p.add_argument("--messages-field", default="messages", help="dotted path of the conversation list")
    p.add_argument("--role-field", default="role")
    p.add_argument("--text-field", default="content")
    p.add_argument("--user-roles", default="user,human")
    p.add_argument("--system-roles", default="system")
    p.add_argument("--strings", choices=["user", "alternate"], default="user",
                   help="conversation given as strings: all user turns, or alternating user/assistant")
    p.add_argument("--id-field", default=None, help="record id field (default: line number)")
    p.add_argument("--cluster-field", default=None, help="field grouping related conversations (e.g. a shared goal id)")
    p.add_argument("--max-user-turns", type=int, default=0, help="truncate after this many user turns (0 = keep all)")
    p.add_argument("--min-user-turns", type=int, default=2, help="skip conversations with fewer user turns")
    args = p.parse_args()
    if bool(args.label) == bool(args.label_field):
        raise SystemExit("give exactly one of --label / --label-field")
    args.user_roles = {r.strip().lower() for r in args.user_roles.split(",")}
    args.system_roles = {r.strip().lower() for r in args.system_roles.split(",")}
    harmful_values = {v.strip().lower() for v in args.harmful_values.split(",")}

    rows, skipped, seen = [], Counter(), set()
    for k, rec in enumerate(read_records(Path(args.input))):
        turns = convert_turns(get(rec, args.messages_field), args)
        n_user = sum(t["role"] == "user" for t in turns)
        if n_user < args.min_user_turns:
            skipped["too_few_user_turns"] += 1
            continue
        rid = str(get(rec, args.id_field)) if args.id_field else str(k)
        eid = f"{args.benchmark}-{rid}"
        if eid in seen:
            skipped["duplicate_id"] += 1
            continue
        seen.add(eid)
        label = args.label or ("harmful" if str(get(rec, args.label_field)).strip().lower() in harmful_values else "benign")
        cluster = get(rec, args.cluster_field) if args.cluster_field else None
        rows.append({"example_id": eid, "benchmark": args.benchmark, "label": label, "turns": turns,
                     "metadata": {"cluster_id": f"{args.benchmark}-c-{cluster}" if cluster is not None else eid,
                                  "source": args.input, "source_id": rid, "n_user_turns": n_user}})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"written": len(rows), "by_label": dict(Counter(r["label"] for r in rows)),
                      "clusters": len({r["metadata"]["cluster_id"] for r in rows}),
                      "user_turns": dict(sorted(Counter(r["metadata"]["n_user_turns"] for r in rows).items())),
                      "skipped": dict(skipped)}, indent=2))


if __name__ == "__main__":
    main()
