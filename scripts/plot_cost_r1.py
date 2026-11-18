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
    p.add_argument("--layout", choices=["full", "compact"], default="full",
                   help="compact: one row per full path (the scorer-only rows, which differ by the controller "
                        "stages, stay in the tables), short labels, panel (b) grouped by scorer, and a fixed "
                        "--width so that the figure is printed at 1:1 (6.5 in = text width of the CMC template)")
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
    base = SCORER_NAMES.get(kind, kind or "scorer")
    return base + " P(unsafe)" if scorer_meta.get("score_mode") == "prob" else base


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


SHORT_SCORER = {"DistilBART-MNLI": "DistilBART", "DeBERTa-v3-large": "DeBERTa",
                "Llama-Guard-3-1B P(unsafe)": "Llama-Guard P(unsafe)", "Llama-Guard-3-1B": "Llama-Guard"}


def compact_rows(latency):
    """Rows of the compact panel (a), top to bottom: (label, [latency records])."""
    rows = []

    def add(label, recs):
        if recs:
            rows.append((label, recs))

    add("Static prompt patch", [r for r in latency if r["key"] == "static"])
    add("RCDP controller stages", [r for r in latency if r["key"] == "routing"])
    scorers = []
    for r in latency:
        if r["key"] == "rcdp_path":
            s = r["row"].split(": ", 1)[1]
            if s not in scorers:
                scorers.append(s)
    for s in scorers:
        add("RCDP path, " + SHORT_SCORER.get(s, s), [r for r in latency if r["key"] == "rcdp_path" and r["row"].endswith(s)])
    add("Llama-Guard one-shot verdict", [r for r in latency if r["key"] == "oneshot"])
    sm = [r for r in latency if r["key"] == "smoothllm"]
    add(sm[0]["row"].replace("SmoothLLM-style + LG", "SmoothLLM-style vote") if sm else "", sm)
    add("Hardened output check", [r for r in latency if r["key"] == "output_guard"])
    return rows


def compact_tokens(tokens, measure):
    """Panel (b) of the compact layout: static instruction once, then (all requests, generated) per scorer."""
    key_all, key_gen, key_static = (
        ("hidden_prompt_tokens_mean_all_requests", "hidden_prompt_tokens_mean_generated", "hidden_prompt_tokens_mean")
        if measure == "hidden" else
        ("added_input_tokens_mean_all_requests", "added_input_tokens_mean_generated", "added_input_tokens_mean"))
    statics = {round(t["static_patch"][key_static], 6) for t in tokens if t["static_patch"].get(key_static) is not None}
    if len(statics) > 1:
        raise SystemExit(f"static-instruction token counts differ between inputs: {sorted(statics)}")
    groups = [("Static instruction", None, statics.pop() if statics else None)]
    for t in tokens:
        groups.append((SHORT_SCORER.get(t["scorer"], t["scorer"]), t["rcdp"].get(key_all), t["rcdp"].get(key_gen)))
    return groups


def render_compact(path, latency, tokens, args):
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fs = args.font_size
    small = fs - 0.5
    plt.rcParams.update({"font.size": fs, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                         "ytick.color": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": small})
    devices = []
    for r in latency:
        if r["device"] not in devices:
            devices.append(r["device"])
    style = {d: (SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]) for i, d in enumerate(devices)}
    rows = compact_rows(latency)
    groups = compact_tokens(tokens, args.token_measure) if tokens else []

    W, H = args.width, args.height
    fig = plt.figure(figsize=(W, H))

    def box(x0, y0, w, h):  # inches -> figure fractions; no tight bbox, so the saved size is exactly W x H
        return [x0 / W, y0 / H, w / W, h / H]

    bottom, top = 0.8, 0.2
    a_x0 = 1.78
    a_w = (4.18 - a_x0) if groups else (W - a_x0 - 0.05)
    ax = fig.add_axes(box(a_x0, bottom, a_w, H - bottom - top))
    n = len(rows)
    vals = [v for _, recs in rows for r in recs for v in (r[args.stat], r["mean_ci95_lo"], r["mean_ci95_hi"],
                                                           r["p50"], r["p99"]) if v is not None and v > 0]
    lo_lim, hi_lim = min(vals) / 3.0, max(vals) * 6.0
    ax.set_xscale("log")
    ax.set_xlim(lo_lim, hi_lim)
    pt_per_decade = a_w * 72.0 / (math.log10(hi_lim) - math.log10(lo_lim))
    for i, (label, recs) in enumerate(rows):
        y0 = n - 1 - i
        for k, r in enumerate(recs):
            m = len(recs)
            y = y0 + (0.0 if m == 1 else (0.2 - 0.4 * k / (m - 1)))
            color, marker = style[r["device"]]
            x = r[args.stat]
            lo, hi = ((r["mean_ci95_lo"], r["mean_ci95_hi"]) if args.stat == "mean" else
                      (r["p50_ci95_lo"], r["p50_ci95_hi"]))
            ax.plot([r["p50"], r["p99"]], [y, y], color=color, lw=0.9, alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot([r["p99"]], [y], marker="|", color=color, ms=4.0, mew=0.9, alpha=0.8, zorder=2)
            if lo is not None and hi is not None:
                ax.errorbar([x], [y], xerr=[[max(x - lo, 0.0)], [max(hi - x, 0.0)]], fmt="none", ecolor=color,
                            elinewidth=1.0, capsize=1.8, zorder=3)
            ax.plot([x], [y], marker=marker, ms=4.2, color=color, mec="white", mew=0.5, zorder=4)
            text = fmt_ms(x)
            text_dec = len(text) * small * 0.56 / pt_per_decade  # label width in decades
            right = max(v for v in (x, hi, r["p99"]) if v is not None)
            left = min(v for v in (x, lo, r["p50"]) if v is not None and v > 0)
            if math.log10(right * 1.4) + text_dec < math.log10(hi_lim):
                ax.text(right * 1.4, y, text, va="center", ha="left", fontsize=small, color=INK_2)
            else:  # no room on the right: label to the left of the interval
                ax.text(left / 1.4, y, text, va="center", ha="right", fontsize=small, color=INK_2)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([lab for lab, _ in rows])
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlabel("ms per request (log scale)", labelpad=1.0)
    ax.set_title("(a) Per-request latency, batch size 1", loc="left", fontsize=fs, pad=3)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", pad=1.5)
    handles = [Line2D([], [], color=style[d][0], marker=style[d][1], ls="", ms=4.2, mec="white", label=d)
               for d in devices]
    handles.append(Line2D([], [], color=INK_2, lw=0.9, alpha=0.5, marker="|", ms=4.0,
                          label=f"{'mean' if args.stat == 'mean' else 'p50'}, 95% CI; line p50 to p99"))
    fig.legend(handles=handles, loc="lower left", ncol=1 if groups else len(handles), frameon=False,
               bbox_to_anchor=(0.0, 0.0) if not groups else (0.0, -0.01), handletextpad=0.3,
               columnspacing=1.0, borderaxespad=0.1, labelspacing=0.1)
    if groups:
        # with panel (b) the device legend is stacked at the lower left, under the y labels of panel (a)
        ax.set_position(box(a_x0, bottom, a_w, H - bottom - top))

    if groups:
        tok_colors = (SLOTS[len(devices) % len(SLOTS)], SLOTS[(len(devices) + 1) % len(SLOTS)])
        b_x0 = 5.02
        bx = fig.add_axes(box(b_x0, bottom, W - b_x0 - 0.04, H - bottom - top))
        g = len(groups)
        vmax = max(v for _, a, b in groups for v in (a, b) if v is not None)
        for i, (label, v_all, v_gen) in enumerate(groups):
            y0 = g - 1 - i
            if v_all is None:  # static instruction: the same tokens on every request
                bx.barh([y0], [v_gen], height=0.42, color=GRID, edgecolor=INK_2, lw=0.5)
                bx.text(v_gen, y0, f" {v_gen:.1f}", va="center", ha="left", fontsize=small, color=INK_2)
                continue
            for j, (v, c) in enumerate(((v_all, tok_colors[0]), (v_gen, tok_colors[1]))):
                y = y0 + (0.17 if j == 0 else -0.17)
                bx.barh([y], [v], height=0.32, color=c, edgecolor="white", lw=0.6)
                bx.text(v, y, f" {v:.1f}", va="center", ha="left", fontsize=small, color=INK_2)
        bx.set_yticks([g - 1 - i for i in range(g)])
        bx.set_yticklabels([lab.replace(" P(unsafe)", "") .replace("Static instruction", "Static") for lab, _, _ in groups])
        bx.set_xlim(0, vmax * 1.3)
        bx.set_ylim(-0.55, g - 0.45)
        bx.set_xlabel("tokens per request", labelpad=1.0)
        bx.set_title("(b) Hidden instruction" if args.token_measure == "hidden" else "(b) Added input tokens",
                     loc="left", fontsize=fs, pad=3)
        bx.grid(axis="x", color=GRID, lw=0.5)
        bx.set_axisbelow(True)
        for side in ("top", "right"):
            bx.spines[side].set_visible(False)
        bx.tick_params(axis="y", length=0, pad=2)
        bx.tick_params(axis="x", pad=1.5)
        fig.legend(handles=[Patch(color=tok_colors[0], label="RCDP, all requests"),
                            Patch(color=tok_colors[1], label="RCDP, sent to the model"),
                            Patch(facecolor=GRID, edgecolor=INK_2, lw=0.5, label="static, every request")],
                   loc="lower left", bbox_to_anchor=((b_x0 - 0.78) / W, -0.01), ncol=1, frameon=False,
                   handlelength=1.0, handletextpad=0.3, labelspacing=0.1, borderaxespad=0.1)
    fig.savefig(path, pad_inches=0.0)
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
    if args.layout == "compact":
        render_compact(out, latency, tokens, args)
    else:
        render(out, latency, bars, args)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
