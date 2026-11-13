"""Fig. 5 (serving_cost_overhead.pdf) from measured latency files (revision R1, ledger R2-4 / R3-5).

Panel (a): per-request latency of the RCDP stages and the alternative safeguards,
one dot per (component, device): the mean with its 95% bootstrap interval, and a
thin line from p50 to p99, on a log axis. Panel (b): hidden-prompt tokens per
request for the static patch and for RCDP (all requests, quarantined ones
counting 0; and generated requests only).

Every plotted value is read from the ``scripts/measure_latency_r1.py measure``
outputs given with ``--inputs``; nothing is typed in. The plotted values, with
their source file and metric, are always written to ``<output stem>.csv`` and
``<output stem>.json``. matplotlib is optional: without it only the tables are
written and the script exits 0.

Devices: dots are coloured by device ("CPU: <model> (<threads> thr)" or
"GPU: <name>", read from each file's recorded environment). Pure-Python stages
(static patch, routing) run on the host CPU whatever the scorer device; when two
inputs give the same (component, device), the first input wins.

Example:
    python scripts/plot_cost_r1.py \
        --inputs results/r1/latency/distilbart_cpu.json results/r1/latency/distilbart_cuda.json \
                 results/r1/latency/deberta_cpu.json results/r1/latency/deberta_cuda.json \
        --output results/r1/figures/serving_cost_overhead.pdf
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

# Categorical slots of the validated reference palette (light mode), fixed order, never cycled.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]  # secondary encoding beyond colour
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SCORER_NAMES = {"distilbart-mnli": "DistilBART-MNLI", "deberta-zeroshot": "DeBERTa-v3-large",
                "llama-guard": "Llama-Guard-3-1B", "keyword": "keyword (smoke test)"}

# (key, metrics in order of preference, label template, device source)
ROWS = [
    ("static", ["static_assembly_ms"], "Static prompt patch", "host"),
    ("routing", ["controller_incremental_replay_ms", "controller_incremental_ms"],
     "RCDP routing (head, Platt, policy)", "host"),
    ("scorer", ["scorer_ms"], "RCDP scorer: {scorer}", "scorer"),
    ("rcdp_path", ["pre_generation_path_ms"], "RCDP full path: {scorer}", "scorer"),
    ("oneshot", ["lg_oneshot_ms"], "Llama Guard 3-1B one-shot", "alt"),
    ("smoothllm", ["smoothllm_ms"], "SmoothLLM-style + LG (N = {copies})", "alt"),
    ("output_guard", ["output_guard_ms"], "Hardened output check (per call)", "output_guard"),
]
EXPECTED = ("static", "routing", "scorer", "rcdp_path", "oneshot", "smoothllm")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Regenerate Fig. 5 from measure_latency_r1.py outputs.")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True, help="figure path (.pdf or .png); tables go next to it")
    p.add_argument("--stat", choices=["mean", "p50"], default="mean", help="dot position in panel (a)")
    p.add_argument("--token-measure", choices=["hidden", "added"], default="hidden",
                   help="panel (b): hidden instruction tokens (paper's Delta_tok) or added templated input tokens")
    p.add_argument("--include-output-guard", action="store_true", help="add the Hardened output check row")
    p.add_argument("--device-label", action="append", default=[],
                   help="override a device label: 'CPU: long name=Xeon 8358' (repeatable)")
    p.add_argument("--width", type=float, default=6.3, help="inches")
    p.add_argument("--height", type=float, default=3.0, help="inches")
    p.add_argument("--font-size", type=float, default=8.5)
    return p.parse_args(argv)


def short_cpu(name):
    s = re.sub(r"\((R|TM|tm|r)\)", "", str(name or "unknown CPU"))
    s = re.sub(r"\bCPU\b|\bProcessor\b|@.*$|\d+-Core", "", s)
    s = re.sub(r"\s+", " ", s).strip() or "unknown CPU"
    return s if len(s) <= 34 else s[:33].rstrip(" ,") + "…"


def short_gpu(name):
    return re.sub(r"^NVIDIA\s+", "", str(name or "GPU")).strip()


def device_label(env, dev):
    dev = str(dev or "cpu")
    if dev.startswith("cuda"):
        idx = int(dev.split(":")[1]) if ":" in dev else 0
        gpus = env.get("gpus") or []
        return "GPU: " + short_gpu(gpus[idx] if idx < len(gpus) else "unknown")
    thr = env.get("torch_num_threads")
    return "CPU: " + short_cpu(env.get("cpu_model")) + (f" ({thr} thr)" if thr else "")


def scorer_name(scorer_meta):
    kind = scorer_meta.get("kind")
    name = str(scorer_meta.get("name") or "")
    if "deberta" in name.lower():
        return SCORER_NAMES["deberta-zeroshot"]
    return SCORER_NAMES.get(kind, kind or "scorer")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(paths, overrides, include_output_guard):
    latency, tokens, missing_notes = [], [], []
    seen_lat, seen_tok = set(), set()
    for path in paths:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = d.get("meta") or {}
        if not str(meta.get("runner", "")).startswith("scripts/measure_latency_r1.py measure"):
            raise SystemExit(f"{path}: not a measure_latency_r1.py 'measure' output")
        env, models, summary = meta.get("environment") or {}, meta.get("models") or {}, d.get("summary") or {}
        sm = models.get("scorer") or {}
        alt = models.get("alt_guard") or {}
        og = models.get("output_guard") or {}
        devs = {"host": device_label(env, "cpu"), "scorer": device_label(env, sm.get("device")),
                "alt": device_label(env, alt.get("device")), "output_guard": device_label(env, og.get("device"))}
        devs = {k: overrides.get(v, v) for k, v in devs.items()}
        sname = scorer_name(sm)
        copies = (alt.get("smoothllm") or {}).get("copies_N")
        boundaries = (d.get("protocol") or {}).get("boundaries") or {}
        src = {"file": Path(path).name, "sha256": sha256_file(path)}
        for key, metrics, template, dev_src in ROWS:
            if key == "output_guard" and not include_output_guard:
                continue
            metric = next((m for m in metrics if summary.get(m, {}).get("n")), None)
            if metric is None:
                continue
            s = summary[metric]
            row = template.format(scorer=sname, copies=copies if copies is not None else "?")
            ident = (row, devs[dev_src])
            if ident in seen_lat:
                continue
            seen_lat.add(ident)
            latency.append({
                "panel": "a", "key": key, "row": row, "device": devs[dev_src], "metric": metric,
                "n": s["n"], "n_calls": s.get("n_calls"), "mean": s["mean"],
                "mean_ci95_lo": (s.get("mean_ci95") or [None, None])[0],
                "mean_ci95_hi": (s.get("mean_ci95") or [None, None])[1],
                "p50": s["p50"], "p50_ci95_lo": (s.get("p50_ci95") or [None, None])[0],
                "p50_ci95_hi": (s.get("p50_ci95") or [None, None])[1],
                "p90": s["p90"], "p99": s["p99"], "boundary": boundaries.get(metric), "source": src,
            })
        t = d.get("token_overhead") or {}
        if "rcdp" in t:
            tk = t.get("tokenizer") or {}
            tok_name = str(tk.get("source") or tk.get("tokenizer_class") or "tokenizer")
            setting = (d.get("protocol") or {}).get("setting")
            ident = (tok_name, sname, setting)
            if ident not in seen_tok:
                seen_tok.add(ident)
                tokens.append({"panel": "b", "tokenizer": tok_name, "scorer": sname, "setting": setting,
                               "n_requests": t.get("n_requests"), "n_generated": t.get("n_generated"),
                               "rcdp": t["rcdp"], "static_patch": t["static_patch"],
                               "configured_budget_mean_all_requests": t.get("configured_budget_mean_all_requests"),
                               "source": src})
    present = {r["key"] for r in latency}
    for key in EXPECTED:
        if key not in present:
            missing_notes.append(f"no input measures '{key}'")
    if not tokens:
        missing_notes.append("no input has token counts (run measure with --prompt-tokenizer)")
    return latency, tokens, missing_notes


def token_bars(tokens, measure):
    """Panel (b) bars; the RCDP configuration is named only when the inputs contain several."""
    bars = []
    several = len({(t["scorer"], t["setting"]) for t in tokens}) > 1
    for t in tokens:
        tag = " | %s, %s" % (t["scorer"], t["setting"]) if several else ""
        if measure == "hidden":
            vals = [("Static patch", t["static_patch"]["hidden_prompt_tokens_mean"]),
                    ("RCDP, all requests" + tag, t["rcdp"]["hidden_prompt_tokens_mean_all_requests"]),
                    ("RCDP, generated" + tag, t["rcdp"]["hidden_prompt_tokens_mean_generated"])]
        else:
            vals = [("Static patch", t["static_patch"]["added_input_tokens_mean"]),
                    ("RCDP, all requests" + tag, t["rcdp"]["added_input_tokens_mean_all_requests"]),
                    ("RCDP, generated" + tag, t["rcdp"]["added_input_tokens_mean_generated"])]
        for label, v in vals:
            if v is not None:
                bars.append({"category": label, "series": t["tokenizer"], "value": v, "source": t["source"]})
    return bars


def write_tables(stem, latency, bars, tokens, notes, args):
    rows = []
    for r in latency:
        rows.append({k: r.get(k) for k in ("panel", "row", "device", "metric", "n", "n_calls", "mean",
                                            "mean_ci95_lo", "mean_ci95_hi", "p50", "p90", "p99")})
        rows[-1]["source_file"] = r["source"]["file"]
    for b in bars:
        rows.append({"panel": "b", "row": b["category"], "device": b["series"],
                     "metric": f"{args.token_measure} tokens per request", "mean": b["value"],
                     "source_file": b["source"]["file"]})
    cols = ["panel", "row", "device", "metric", "n", "n_calls", "mean", "mean_ci95_lo", "mean_ci95_hi",
            "p50", "p90", "p99", "source_file"]
    with open(stem.with_suffix(".csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    payload = {"runner": "scripts/plot_cost_r1.py", "argv": sys.argv[1:], "stat": args.stat,
               "token_measure": args.token_measure, "inputs": [str(p) for p in args.inputs],
               "latency": latency, "token_overhead": tokens, "token_bars": bars, "notes": notes}
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def fmt_ms(x):
    if x >= 100:
        return f"{x:,.0f} ms"
    return f"{x:.3g} ms"


def render(path, latency, bars, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": args.font_size, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK, "text.color": INK, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "font.family": "DejaVu Sans"})
    devices = []
    for r in latency:
        if r["device"] not in devices:
            devices.append(r["device"])
    style = {d: (SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]) for i, d in enumerate(devices)}
    order = []
    for key, _, _, _ in ROWS:
        for r in latency:
            if r["key"] == key and r["row"] not in order:
                order.append(r["row"])

    has_b = bool(bars)
    fig = plt.figure(figsize=(args.width, args.height))
    gs = fig.add_gridspec(1, 2 if has_b else 1, width_ratios=[1.9, 1] if has_b else [1], wspace=0.06,
                          left=0.0, right=1.0, bottom=0.16, top=0.9)
    ax = fig.add_subplot(gs[0, 0])
    ypos = {row: len(order) - 1 - i for i, row in enumerate(order)}
    xs_all = []
    for r in latency:
        devs_in_row = [x["device"] for x in latency if x["row"] == r["row"]]
        k, m = devs_in_row.index(r["device"]), len(devs_in_row)
        y = ypos[r["row"]] + (0.0 if m == 1 else (0.22 - 0.44 * k / (m - 1)))
        color, marker = style[r["device"]]
        x = r[args.stat]
        lo, hi = (r["mean_ci95_lo"], r["mean_ci95_hi"]) if args.stat == "mean" else (r["p50_ci95_lo"], r["p50_ci95_hi"])
        ax.plot([r["p50"], r["p99"]], [y, y], color=color, lw=1.0, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot([r["p99"]], [y], marker="|", color=color, ms=6, mew=1.2, alpha=0.8, zorder=2)
        if lo is not None and hi is not None:
            ax.errorbar([x], [y], xerr=[[max(x - lo, 0.0)], [max(hi - x, 0.0)]], fmt="none", ecolor=color,
                        elinewidth=1.4, capsize=2.5, zorder=3)
        ax.plot([x], [y], marker=marker, ms=5.5, color=color, mec="white", mew=0.8, zorder=4)
        right = max(v for v in (x, hi, r["p99"]) if v is not None)
        ax.text(right * 1.35, y, fmt_ms(x), va="center", ha="left", fontsize=args.font_size - 1, color=INK_2)
        xs_all.extend(v for v in (x, lo, hi, r["p50"], r["p99"]) if v is not None and v > 0)
    ax.set_xscale("log")
    if xs_all:
        ax.set_xlim(min(xs_all) / 3.0, max(xs_all) * 15.0)
    ax.set_yticks([ypos[r] for r in order])
    ax.set_yticklabels(order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("ms per request (log scale)")
    ax.set_title("(a) Per-request latency, batch 1", loc="left", fontsize=args.font_size + 0.5)
    ax.grid(axis="x", color=GRID, lw=0.6, which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=style[d][0], marker=style[d][1], ls="", ms=5.5, mec="white", label=d)
               for d in devices]
    handles.append(Line2D([], [], color=INK_2, lw=1.0, alpha=0.5, marker="|", ms=6,
                          label=f"{'mean' if args.stat == 'mean' else 'p50'} with 95% CI; line p50 to p99"))

    if has_b:
        bx = fig.add_subplot(gs[0, 1])
        series = []
        for b in bars:
            if b["series"] not in series:
                series.append(b["series"])
        cats = []
        for b in bars:
            if b["category"] not in cats:
                cats.append(b["category"])
        h = 0.55 / max(len(series), 1)
        for b in bars:
            si = series.index(b["series"])
            y = len(cats) - 1 - cats.index(b["category"]) + (0.0 if len(series) == 1 else (0.275 - h / 2 - si * h))
            bx.barh([y], [b["value"]], height=h * 0.9, color=SLOTS[si % len(SLOTS)], edgecolor="white", lw=1.0)
            bx.text(b["value"], y, f" {b['value']:.1f}", va="center", ha="left", fontsize=args.font_size - 1,
                    color=INK_2)
        bx.set_yticks([len(cats) - 1 - i for i in range(len(cats))])
        bx.set_yticklabels(cats)
        bx.yaxis.tick_right()
        bx.tick_params(axis="y", length=0)
        vmax = max(b["value"] for b in bars)
        bx.set_xlim(0, vmax * 1.3 if vmax > 0 else 1)
        bx.set_ylim(-0.6, len(cats) - 0.4)
        bx.set_xlabel("tokens per request")
        bx.set_title("(b) Hidden-prompt tokens" if args.token_measure == "hidden" else "(b) Added input tokens",
                     loc="left", fontsize=args.font_size + 0.5)
        bx.grid(axis="x", color=GRID, lw=0.6)
        bx.set_axisbelow(True)
        for side in ("top", "left"):
            bx.spines[side].set_visible(False)
        if len(series) > 1:
            bx.legend(handles=[Line2D([], [], color=SLOTS[i], lw=6, label=s) for i, s in enumerate(series)],
                      frameon=False, loc="lower right", fontsize=args.font_size - 1, title="tokenizer")
    # Legend below the axes (outside them); bbox_inches="tight" keeps it in the saved figure.
    fig.legend(handles=handles, loc="upper left", ncol=1 if len(handles) > 3 else len(handles), frameon=False,
               fontsize=args.font_size - 1, bbox_to_anchor=(ax.get_position().x0, 0.0))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    overrides = {}
    for item in args.device_label:
        if "=" not in item:
            raise SystemExit(f"--device-label needs 'detected=replacement', got {item!r}")
        k, v = item.rsplit("=", 1)
        overrides[k.strip()] = v.strip()
    latency, tokens, notes = collect(args.inputs, overrides, args.include_output_guard)
    if not latency:
        raise SystemExit("no latency summaries found in the inputs")
    bars = token_bars(tokens, args.token_measure)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    write_tables(stem, latency, bars, tokens, notes, args)
    for n in notes:
        print("note:", n)
    for r in latency:
        print(f"(a) {r['row']:48s} {r['device']:40s} mean {r['mean']:.4g} ms  p50 {r['p50']:.4g}  "
              f"p99 {r['p99']:.4g}  n={r['n']}")
    for b in bars:
        print(f"(b) {b['category']:48s} {b['series']:40s} {b['value']:.2f} tokens")
    print("wrote", stem.with_suffix(".csv"), "and", stem.with_suffix(".json"))
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed: figure skipped (tables written)")
        return 0
    render(out, latency, bars, args)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
