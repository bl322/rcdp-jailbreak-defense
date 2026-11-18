# -*- coding: utf-8 -*-
"""Revision R1 cost numbers, recounted from the released measurement files.

Inputs (read-only):
  _r1_out/latency_rerun_20261118/results/r1/cost/latency_rerun_20261118/{distilbart,deberta,llamaguard_prob}_{cpu,cuda}.json
      measure_latency_r1.py outputs: the six configurations, measured one after another in one session (2026-11-18)
  ccfa-workfiles/tables/cost/build/fig5_rerun_20261118/serving_cost_overhead.csv
      plot_cost_r1.py output (Fig. 5, --stat p50); checked when present
  end-to-end runs (per-record latency_ms): hosted model from e2e_rcdp, local models from e2e_r3

Outputs: build/cost_r1.{md,json} and build/latency_rows.tex (appendix table body).

    python -I -X utf8 verify_cost.py [--root <project root>]
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAT = "_r1_out/latency_rerun_20261118/results/r1/cost/latency_rerun_20261118/"
FIG = "ccfa-workfiles/tables/cost/build/fig5_rerun_20261118/"
SCORERS = ("distilbart", "deberta", "llamaguard_prob")
E2E = {"qwen3-next-80b-a3b-instruct": "_r1_out/e2e_rcdp/extracted/results/r1/e2e/rcdp_qwen_dist_agnostic.json",
       "Llama-3.1-8B-Instruct": "_r1_out/e2e_r3/loc_llama31_rcdp_dist.json",
       "Mistral-7B-Instruct-v0.3": "_r1_out/e2e_r3/loc_mistral_rcdp_dist.json"}


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def q(xs, p):
    """Linear-interpolation quantile (numpy default)."""
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def stats(xs):
    return {"n": len(xs), "mean": sum(xs) / len(xs), "p50": q(xs, 0.5), "p90": q(xs, 0.9), "p99": q(xs, 0.99)}


def lat_n(d, m):
    return d["summary"][m]["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[3]))
    root = Path(ap.parse_args().root)
    out, L = {"latency": {}}, ["# R1 cost numbers (recounted)", ""]
    files = {}
    for sc in SCORERS:
        for dev in ("cpu", "cuda"):
            d = j(root / LAT / f"{sc}_{dev}.json")
            files[(sc, dev)] = d
            env, prot = d["meta"]["environment"], d["protocol"]
            assert prot["batch_size"] == 1 and prot["concurrency"] == 1 and prot["warmup_requests"] == 10
            assert prot["repeats"] == 3 and d["counts"]["requests"] == 612 and d["counts"]["examples"] == 204
            assert "cleared before every request" in prot["cache"]
            assert d["instrumented_vs_plain_decision_mismatches"] == 0
            assert d["calibration"]["protocol"] == "r1" and d["calibration"]["seed"] == "rcdp-r1"
            out["latency"][f"{sc}_{dev}"] = {k: {m: v[m] for m in ("n", "mean", "p50", "p90", "p99")}
                                             for k, v in d["summary"].items() if isinstance(v, dict) and "p50" in v}
            out[f"env_{sc}_{dev}"] = {"cpu": env["cpu_model"], "threads": env["torch_num_threads"],
                                      "gpus": env.get("gpus"), "torch": env["torch"],
                                      "transformers": env["transformers"], "python": env["python"],
                                      "scorer_device": d["meta"]["models"]["scorer"]["device"]}
    e = out["env_distilbart_cpu"]
    assert e["cpu"] == "AMD EPYC 7282 16-Core Processor" and e["threads"] == 16
    assert all(out[f"env_{s}_cuda"]["gpus"] == ["NVIDIA A100-PCIE-40GB"] for s in SCORERS)
    assert all(out[f"env_{s}_cpu"]["cpu"] == e["cpu"] and out[f"env_{s}_cpu"]["threads"] == 16 for s in SCORERS)
    assert {out[f"env_{s}_{d}"]["torch"] for s in SCORERS for d in ("cpu", "cuda")} == {"2.9.1+cu128"}
    # one code state for all six runs
    cv = {files[k]["meta"]["code_version"]["sha256"] for k in files}
    assert len(cv) == 1, cv
    out["code_version"] = cv.pop()
    # alternatives: timed in every pass of the DistilBART runs (51 and 12 prompts on the CPU, all 204 on the GPU)
    for dev, n1, n2 in (("cpu", 51, 12), ("cuda", 204, 204)):
        d = files[("distilbart", dev)]
        assert d["protocol"]["alternative_repeats"] == 3
        assert lat_n(d, "lg_oneshot_ms") == 3 * n1 and lat_n(d, "smoothllm_ms") == 3 * n2, dev
    # The same calibrated controller in all four files (thresholds of the DistilBART files equal Table 1's gate).
    th = files[("distilbart", "cpu")]["calibration"]["thresholds"]
    assert th == files[("distilbart", "cuda")]["calibration"]["thresholds"] and th["tau_q"] == 0.69
    rs = files[("distilbart", "cpu")]["routing_state_counts_rep0"]
    assert rs == {"benign/cautious": 15, "benign/hardened": 13, "benign/quarantine": 27, "harmful/cautious": 10,
                  "harmful/hardened": 20, "harmful/quarantine": 119}
    out["routing_dist_rep0"] = rs

    # tokens
    t = files[("distilbart", "cpu")]["token_overhead"]
    tk = t["tokenizer"]
    assert tk["source"] == "llama-guard-3-1b" and tk["state_instruction_tokens"] == {"normal": 22, "cautious": 35, "hardened": 47}
    assert tk["static_patch_tokens"] == 47 and t["n_requests"] == 204 and t["n_generated"] == 58
    assert t["rcdp"]["hidden_prompt_tokens_sum"] == 25 * 35 + 33 * 47 == 2426
    out["tokens_dist"] = {"state": tk["state_instruction_tokens"], "static": 47, "all": t["rcdp"]["hidden_prompt_tokens_mean_all_requests"],
                          "generated": t["rcdp"]["hidden_prompt_tokens_mean_generated"], "n_generated": 58,
                          "user_tokens_mean": t["user_tokens_mean"]}
    td = files[("deberta", "cpu")]["token_overhead"]
    out["tokens_deb"] = {"all": td["rcdp"]["hidden_prompt_tokens_mean_all_requests"],
                         "generated": td["rcdp"]["hidden_prompt_tokens_mean_generated"], "n_generated": td["n_generated"]}

    # figure table = latency files
    nchk = 0
    if (root / FIG / "serving_cost_overhead.csv").is_file():
        rows = list(csv.DictReader(open(root / FIG / "serving_cost_overhead.csv", encoding="utf-8")))
        srcmap = {f"{sc}_{dev}.json": (sc, dev) for sc in SCORERS for dev in ("cpu", "cuda")}
        for r in rows:
            if r["panel"] != "a":
                continue
            s = out["latency"]["%s_%s" % srcmap[r["source_file"]]][r["metric"]]
            for m in ("mean", "p50", "p90", "p99"):
                assert abs(float(r[m]) - s[m]) < 1e-9, (r["row"], m)
            nchk += 1
        assert nchk == 18, nchk
        # printed only, so that the outputs do not depend on whether the figure build is present
        print(f"figure rows checked against the latency files: {nchk}; figure sha256 "
              f"{hashlib.sha256((root / FIG / 'serving_cost_overhead.pdf').read_bytes()).hexdigest()}")
    else:
        print("figure build not present: figure check skipped")

    # end-to-end components recorded in the runs (per call)
    out["e2e"] = {}
    for model, f in E2E.items():
        d = j(root / f)
        gen = [r["latency_ms"]["generation"] for r in d["records"] if r["generated"]]
        og = [r["latency_ms"]["output_guard"] for r in d["records"] if r.get("output_guard_checked")]
        assert len(d["records"]) == 204 and len(gen) == 58, (model, len(gen))
        assert len(og) == 33 and all(x > 0 for x in og), (model, len(og))
        out["e2e"][model] = {"file": Path(f).name, "generation_per_call": stats(gen), "output_check_per_call": stats(og),
                             "max_new_tokens": d.get("generation_settings", {}).get("max_new_tokens")}

    lat = out["latency"]
    def g(k, m, s="mean"):
        return lat[k][m][s]
    def p50(k, m):
        return lat[k][m]["p50"]
    path = "pre_generation_path_ms"
    out["text"] = {
        "median_path_ms": {f"{sc}_{dev}": p50(f"{sc}_{dev}", path) for sc in SCORERS for dev in ("cpu", "cuda")},
        "median_oneshot_ms": {dev: p50(f"distilbart_{dev}", "lg_oneshot_ms") for dev in ("cpu", "cuda")},
        "median_smoothllm_ms": {dev: p50(f"distilbart_{dev}", "smoothllm_ms") for dev in ("cpu", "cuda")},
        "replay_cpu": {s: lat["distilbart_cpu"]["controller_incremental_replay_ms"][s] for s in ("p50", "p90", "p99")},
        "oneshot_over_path_median": {f"{sc}_{dev}": p50(f"distilbart_{dev}", "lg_oneshot_ms") / p50(f"{sc}_{dev}", path)
                                     for sc in SCORERS for dev in ("cpu", "cuda")},
        "deberta_over_distilbart_median": {dev: p50(f"deberta_{dev}", path) / p50(f"distilbart_{dev}", path)
                                           for dev in ("cpu", "cuda")},
    }
    # DistilBART on the CPU: requests above 1 s (other load on the shared host during the run)
    recs = files[("distilbart", "cpu")]["records"]
    tpath = [(r, r[path]) for r in recs]
    slow = [(r, t) for r, t in tpath if t > 1000]
    out["text"]["distilbart_cpu_over_1s"] = {"requests": len(slow), "of": len(recs),
                                             "prompts": len({r["example_id"] for r, _ in slow}),
                                             "per_pass": [sum(1 for r, _ in slow if r["rep"] == k) for k in range(3)],
                                             "max_ms": max((t for _, t in slow), default=None)}
    L += [f"- code version (sha256 over rcdp_guard and the runner): {out['code_version']}",
          "- medians (full path, ms): " + ", ".join(f"{k} {v:.1f}" for k, v in out["text"]["median_path_ms"].items()),
          f"- medians one-shot {out['text']['median_oneshot_ms']}, SmoothLLM {out['text']['median_smoothllm_ms']}",
          f"- controller replay CPU p50/p90/p99 {out['text']['replay_cpu']}",
          "- one-shot / path (medians): " + ", ".join(f"{k} {v:.2f}" for k, v in out["text"]["oneshot_over_path_median"].items()),
          f"- DeBERTa / DistilBART path (medians): {out['text']['deberta_over_distilbart_median']}",
          f"- DistilBART CPU requests above 1 s: {out['text']['distilbart_cpu_over_1s']}"]
    L += [f"- environment: {e['cpu']} ({e['threads']} threads); GPU {out['env_distilbart_cuda']['gpus'][0]}; torch "
          f"{e['torch']}, transformers {e['transformers']}, Python {e['python']}",
          f"- controller stages (replay, CPU): mean {g('distilbart_cpu', 'controller_incremental_replay_ms'):.4f} ms, p99 "
          f"{g('distilbart_cpu', 'controller_incremental_replay_ms', 'p99'):.4f}; instrumented incremental "
          f"{g('distilbart_cpu', 'controller_incremental_ms'):.4f}",
          f"- DistilBART path CPU {g('distilbart_cpu', 'pre_generation_path_ms'):.1f} GPU {g('distilbart_cuda', 'pre_generation_path_ms'):.1f}; "
          f"DeBERTa path CPU {g('deberta_cpu', 'pre_generation_path_ms'):.1f} GPU {g('deberta_cuda', 'pre_generation_path_ms'):.1f}",
          f"- LG one-shot CPU {g('distilbart_cpu', 'lg_oneshot_ms'):.1f} (n {lat['distilbart_cpu']['lg_oneshot_ms']['n']}) GPU "
          f"{g('distilbart_cuda', 'lg_oneshot_ms'):.1f}; SmoothLLM CPU {g('distilbart_cpu', 'smoothllm_ms'):.0f} (n "
          f"{lat['distilbart_cpu']['smoothllm_ms']['n']}) GPU {g('distilbart_cuda', 'smoothllm_ms'):.1f}",
          f"- CPU ratio one-shot / DistilBART path {g('distilbart_cpu', 'lg_oneshot_ms') / g('distilbart_cpu', 'pre_generation_path_ms'):.2f}",
          f"- tokens: {out['tokens_dist']}; DeBERTa {out['tokens_deb']}"]
    for model, v in out["e2e"].items():
        L.append(f"- e2e {model}: generation per call p50 {v['generation_per_call']['p50']:.0f} p90 "
                 f"{v['generation_per_call']['p90']:.0f} (n {v['generation_per_call']['n']}); output check per call p50 "
                 f"{v['output_check_per_call']['p50']:.1f} p90 {v['output_check_per_call']['p90']:.1f} "
                 f"(n {v['output_check_per_call']['n']})")

    # appendix table body: component | CPU mean p50 p90 p99 | GPU mean p50 p90 p99
    def cell(k, m):
        s = lat[k][m]
        f = (lambda x: f"{x:,.0f}") if s["mean"] >= 1000 else ((lambda x: f"{x:,.1f}") if s["mean"] >= 10 else (lambda x: f"{x:.3f}"))
        return " & ".join(f(s[x]) for x in ("mean", "p50", "p90", "p99"))
    body = [
        ("Features, risk head and Platt map", "distilbart", "features_head_platt_ms"),
        ("Uncertainty, trajectory terms and state selection", "distilbart", "policy_router_ms"),
        ("Controller stages, uninstrumented replay", "distilbart", "controller_incremental_replay_ms"),
        ("Scorer: DistilBART-MNLI", "distilbart", "scorer_ms"),
        ("Full path: DistilBART configuration", "distilbart", "pre_generation_path_ms"),
        ("Scorer: DeBERTa-v3-large", "deberta", "scorer_ms"),
        ("Full path: DeBERTa configuration", "deberta", "pre_generation_path_ms"),
        (r"Scorer: Llama-Guard-3-1B, $P(\texttt{unsafe})$", "llamaguard_prob", "scorer_ms"),
        ("Full path: Llama-Guard configuration", "llamaguard_prob", "pre_generation_path_ms"),
        ("Llama-Guard-3-1B one-shot verdict$^{a}$", "distilbart", "lg_oneshot_ms"),
        ("SmoothLLM-style, 10 Llama-Guard calls$^{a}$", "distilbart", "smoothllm_ms"),
    ]
    tex = []
    for label, sc, m in body:
        tex.append(f"{label} & {cell(sc + '_cpu', m)} & {cell(sc + '_cuda', m)} \\\\")
    (HERE.parent / "build").mkdir(exist_ok=True)
    (HERE.parent / "build" / "latency_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (HERE.parent / "build" / "cost_r1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (HERE.parent / "build" / "cost_r1.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\n".join(L))
    print("\n".join(tex))


if __name__ == "__main__":
    main()
