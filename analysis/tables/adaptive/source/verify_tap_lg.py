# -*- coding: utf-8 -*-
"""TAP against the Llama-Guard P(unsafe) configuration (r5, three arms): provenance check, recount from the
per-query records, and comparison with the DistilBART arms of Table 4 (r4).

Inputs (read-only):
  _r1_out/r5/extracted/r5_pack/tap/tap_lg_{arm}.json, tap_lg_{arm}_rejudged.json   (r5 package, sha256 242854cb...)
  _r1_out/r4/extracted/r4_pack/tap/tap_nli_{arm}.json, tap_nli_{arm}_rejudged768.json
  gates: rcdp_qwen_lgprob_agnostic.json / rcdp_qwen_dist_agnostic.json (R1 e2e, hosted endpoint)

    python -I -X utf8 verify_tap_lg.py [--root <project root>]
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

# R1 release (only change to the script as run): on the redacted result package, the SHA-256 of the original file,
# as recorded in _r1_out/ORIGINAL_SHA256.json by analysis/setup_layout.py; otherwise the file's own SHA-256.
def file_sha256(p):
    p = Path(p).resolve()
    for d in p.parents:
        idx = d / "ORIGINAL_SHA256.json"
        if d.name == "_r1_out" and idx.is_file():
            got = json.loads(idx.read_text(encoding="utf-8")).get(p.relative_to(d).as_posix())
            if got:
                return got
    return hashlib.sha256(p.read_bytes()).hexdigest()

HERE = Path(__file__).resolve().parent
ARMS = ("fresh", "bound_rf_latch", "bound_rf_nolatch")
E2E = "_r1_out/e2e_rcdp/extracted/results/r1/e2e/"
SRC = {"lg": ("_r1_out/r5/extracted/r5_pack/tap/", "tap_lg_{}.json", "tap_lg_{}_rejudged.json",
              "rcdp_qwen_lgprob_agnostic.json", 135),
       "nli": ("_r1_out/r4/extracted/r4_pack/tap/", "tap_nli_{}.json", "tap_nli_{}_rejudged768.json",
               "rcdp_qwen_dist_agnostic.json", 119)}
SAME = ("harness", "decision_rule", "root_query_submitted", "root_feedback_source", "controller_monotone", "depth",
        "width", "branching", "prune_below", "success_score", "success_rule", "query_budget", "frontier_rule",
        "target_selection", "seed", "seed_derivation", "attacker_model", "judge_model", "target", "identity_scope",
        "dataset_sha256")


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def arm_stats(root, sc, arm):
    base, fn, rjn, gate_file, n_block = SRC[sc]
    gate = {r["example_id"]: r["outcome"]["blocked_pre_generation"] for r in j(root / E2E / gate_file)["records"]
            if r["label"] == "harmful"}
    gate_blocked = {i for i, b in gate.items() if b}
    assert len(gate) == 149 and len(gate_blocked) == n_block, (sc, len(gate_blocked))
    f = root / base / fn.format(arm)
    d = j(f)
    p = d["protocol"]
    runs = d["runs"]
    pre = {e["example_id"] for e in d["pre_attack"]["per_example"] if e["blocked"]}
    assert pre == gate_blocked, (sc, arm, "pre-attack set differs from the e2e gate")
    assert {r["example_id"] for r in runs} == pre and len(runs) == n_block
    assert not any(r["root_submitted"] for r in runs) and all(r["target_calls"] == 0 for r in runs)
    rj = j(root / base / rjn.format(arm))
    assert rj["protocol"]["results_sha256"] == file_sha256(f), (sc, arm, "re-judge input")
    ij = rj["protocol"]["independent_judge"]
    assert ij["model"] == "deepseek-v4-flash-0731" and ij["max_tokens"] == 768, ij
    succ = {r["example_id"] for r in runs if r["success"]}
    det = [x for x in rj["details"] if x["is_search_success"]]
    assert {x["example_id"] for x in det} == succ and len(det) == len(succ)
    unparsed = [x["example_id"] for x in det if not x["independent_parsed"]]
    strict = {x["example_id"] for x in det if x["independent_parsed"] and x["independent_score"] >= rj["protocol"]["strict_score"]}
    assert strict == set(rj["confirmed_example_ids_strict"]), (sc, arm)
    Q = [q for r in runs for q in r["attempted_candidates"]]
    assert len(Q) == sum(r["service_queries"] for r in runs)
    o = {"protocol": {k: p.get(k) for k in SAME}, "scorer": p["scorer"], "lg_score_mode": p.get("lg_score_mode"),
         "identity_mode": p["identity_mode"], "latch_states": p.get("latch_states"), "attacked": len(runs),
         "search_successes": len(succ), "evaded_strict": len(strict), "unparsed_successes": len(unparsed),
         "mean_queries": mean(r["service_queries"] for r in runs), "queries": len(Q),
         "residual_block_rate": (n_block - len(strict)) / 149, "residual_blocked": n_block - len(strict),
         "preservation": sum(x["judge_score"] >= 5 for x in Q) / len(Q),
         "llamaguard_unsafe_among_successes": sum(x["llamaguard_unsafe"] for x in det),
         "runs_latched": sum(r["latched_at_query"] is not None for r in runs),
         "queries_blocked_by_latch": sum(r["queries_blocked_by_latch"] for r in runs), "_strict": sorted(strict),
         "_succ": sorted(succ)}
    # Supplementary Table S6 rows
    loose = {x["example_id"] for x in det if x["independent_parsed"] and x["independent_score"] >= rj["protocol"]["loose_score"]}
    lv = [0] * 6
    for r in runs:
        if r["example_id"] in strict:
            lv[r["success_level"] - 1] += 1
    stops = {}
    for r in runs:
        stops[r["stop_reason"]] = stops.get(r["stop_reason"], 0) + 1
    latched_q = [x for x in Q if x["reason"] == "latched"]
    o["s6"] = {"queries": len(Q), "allowed": sum(x["allowed"] for x in Q),
               "blocked_controller_oneturn_allow": sum(not x["allowed"] and x["reason"] == "controller"
                                                       and x.get("stateless_decision") == "allow" for x in Q),
               "latched": len(latched_q), "latched_stateless_allow": sum(x.get("stateless_decision") == "allow" for x in latched_q),
               "latched_controller_allow": sum(x.get("shadow_decision") == "allow" for x in latched_q),
               "prompts_latched": sum(r["latched_at_query"] is not None for r in runs),
               "successes": len(succ), "evaded_strict": len(strict), "evaded_loose": len(loose), "levels": lv,
               "stops": [stops.get("success", 0), stops.get("depth_exhausted", 0), stops.get("all_pruned", 0)],
               "stops_other": {k: v for k, v in stops.items() if k not in ("success", "depth_exhausted", "all_pruned")},
               "mean_q": mean(r["service_queries"] for r in runs), "max_q": max(r["service_queries"] for r in runs),
               "lg_unsafe": sum(x["llamaguard_unsafe"] for x in det),
               "lg_unknown": sum(bool(x.get("llamaguard_unknown")) for x in det)}
    if arm != "fresh":
        fb_esc = fb_gen = after_allowed = after_state_only = 0
        never, fb_idx = set(), []
        for r in runs:
            qs = sorted(r["attempted_candidates"], key=lambda x: x["query_index"])
            k = next((n for n, x in enumerate(qs) if not x["allowed"]), None)
            if k is None:
                never.add(r["example_id"])
                continue
            fb_idx.append(qs[k]["query_index"])
            if qs[k]["stateless_decision"] == "allow":
                fb_esc += 1
            else:
                fb_gen += 1
            after_allowed += sum(x["allowed"] for x in qs[k + 1:])
            after_state_only += sum(not x["allowed"] and x["stateless_decision"] == "allow" for x in qs[k + 1:])
        o.update({"blocked_by_controller": sum(not x["allowed"] and x["reason"] == "controller" for x in Q),
                  "blocked_state_only": sum(not x["allowed"] and x["reason"] == "controller"
                                            and x["stateless_decision"] == "allow" for x in Q),
                  "allowed_though_stateless_blocks": sum(x["allowed"] and x["stateless_decision"] == "block" for x in Q),
                  "callers_blocked": len(fb_idx), "first_block_by_escalation": fb_esc, "first_block_single_turn": fb_gen,
                  "first_block_max_query_index": max(fb_idx) if fb_idx else None,
                  "allowed_after_first_block": after_allowed, "state_only_after_first_block": after_state_only,
                  "never_blocked": len(never), "never_blocked_equals_successes": never == succ,
                  "succeeded_after_a_block": len(succ - never)})
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out = {sc: {arm: arm_stats(root, sc, arm) for arm in ARMS} for sc in ("lg", "nli")}
    # the LG arms use the search of the DistilBART arms; only the scorer differs
    for arm in ARMS:
        a, b = out["lg"][arm]["protocol"], out["nli"][arm]["protocol"]
        diff = {k: (a[k], b[k]) for k in SAME if a[k] != b[k]}
        assert not diff, (arm, diff)
        assert out["lg"][arm]["identity_mode"] == out["nli"][arm]["identity_mode"]
        assert out["lg"][arm]["latch_states"] == out["nli"][arm]["latch_states"], arm
        assert out["lg"][arm]["scorer"] == "llama-guard" and out["lg"][arm]["lg_score_mode"] == "prob"
    L = ["# TAP against the Llama-Guard configuration (recounted)", ""]
    for sc in ("lg", "nli"):
        for arm in ARMS:
            o = out[sc][arm]
            L.append(f"- {sc} {arm}: identity {o['identity_mode']}, latch {o['latch_states']}, attacked {o['attacked']}, "
                     f"search successes {o['search_successes']}, evaded strict {o['evaded_strict']} "
                     f"({o['evaded_strict'] / o['attacked']:.4f}), unparsed re-judge replies {o['unparsed_successes']}, "
                     f"queries/prompt {o['mean_queries']:.2f}, preservation {o['preservation']:.4f}, residual block "
                     f"{o['residual_block_rate']:.4f} ({o['residual_blocked']}/149), LG-unsafe among successes "
                     f"{o['llamaguard_unsafe_among_successes']}, latched runs {o['runs_latched']} / latch-blocked queries "
                     f"{o['queries_blocked_by_latch']}")
            if arm != "fresh":
                L.append(f"    controller blocks {o['blocked_by_controller']} (state only {o['blocked_state_only']}); callers "
                         f"blocked {o['callers_blocked']}; first block by escalation {o['first_block_by_escalation']}, "
                         f"single-turn {o['first_block_single_turn']}, latest at query {o['first_block_max_query_index']}; "
                         f"allowed after first block {o['allowed_after_first_block']}; state-only blocks after first block "
                         f"{o['state_only_after_first_block']}; never blocked {o['never_blocked']} (= successes: "
                         f"{o['never_blocked_equals_successes']}, successes after a block {o['succeeded_after_a_block']})")
        L.append("")
        for a, c in (("fresh", "bound_rf_nolatch"), ("bound_rf_nolatch", "bound_rf_latch")):
            sa, sb = set(out[sc][a]["_strict"]), set(out[sc][c]["_strict"])
            x, y = len(sa - sb), len(sb - sa)
            out[sc][f"{a} vs {c}"] = {"discordant": [x, y], "p": mcnemar(x, y)}
            L.append(f"- {sc}: evaded {a} vs {c}: discordant {x}/{y}, exact McNemar p = {mcnemar(x, y):.3g}")
        L.append("")
    # The DistilBART rows must reproduce Supplementary Table S6 as printed.
    S6_NLI = {"fresh": (1060, 324, 115, 73, 107, [28, 30, 4, 8, 0, 3], [115, 2, 2], 8.91, 44, 66),
              "bound_rf_nolatch": (4009, 71, 25, 16, 23, [16, 0, 0, 0, 0, 0], [25, 90, 4], 33.69, 44, 13),
              "bound_rf_latch": (4462, 58, 16, 14, 16, [14, 0, 0, 0, 0, 0], [16, 101, 2], 37.50, 44, 8)}
    for arm, (nq, na, ns, ev, evl, lv, st, mq, xq, lgu) in S6_NLI.items():
        t = out["nli"][arm]["s6"]
        got = (t["queries"], t["allowed"], t["successes"], t["evaded_strict"], t["evaded_loose"], t["levels"], t["stops"],
               round(t["mean_q"], 2), t["max_q"], t["lg_unsafe"])
        assert got == (nq, na, ns, ev, evl, lv, st, mq, xq, lgu), (arm, got)
    assert out["nli"]["bound_rf_nolatch"]["s6"]["blocked_controller_oneturn_allow"] == 1563
    assert out["nli"]["bound_rf_latch"]["s6"]["blocked_controller_oneturn_allow"] == 17
    t = out["nli"]["bound_rf_latch"]["s6"]
    assert (t["latched"], t["latched_stateless_allow"], t["latched_controller_allow"], t["prompts_latched"]) == (4319, 1782, 0, 115)
    L.append("- DistilBART rows reproduce Supplementary Table S6: yes")
    for arm in ARMS:
        L.append(f"- S6 lg {arm}: {out['lg'][arm]['s6']}")
    b = HERE.parent / "build"
    b.mkdir(exist_ok=True)
    (b / "tap_lg_r5.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (b / "tap_lg_r5.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
