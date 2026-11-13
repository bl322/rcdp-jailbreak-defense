"""End-to-end RCDP probe (Table 3, revision R1): routing -> protected model -> judges.

For every held-out item the output records separately
  (a) the controller's pre-generation block (Quarantine),
  (b) the Hardened output-check block (raw generation and its verdict are kept),
  (c) the protected model's refusal (selectable, documented refusal detector),
  (d) the judged harmful completion (primary judge; optional second judge),
  (e) a useful answer for benign items,
and reports ABR_ctrl, ABR_e2e (= a or b or c), 1-ASR, BAR and the benign useful-answer
rate, each as k/n with exact and bootstrap 95% intervals. All generation settings, system
prompts, chat templates, response texts and the calibration report are saved.

Main configuration (abstract / Table 3, benchmark-agnostic):

    python scripts/run_end_to_end_eval_gpu.py \
        --dataset data/paper_splits/heldout_204.jsonl \
        --calibration-dataset data/paper_splits/full_pool_calibration.jsonl \
        --benchmark-mode agnostic \
        --inference-backend local --model-path <LLAMA31_8B_DIR> \
        --response-judge-model-path <LLAMA_GUARD_3_1B_DIR> \
        --output-guard-backend llama-guard --output-guard-model-path <LLAMA_GUARD_3_1B_DIR> \
        --output results/r1/e2e_llama31_agnostic.json

Changes from the submitted harness (audit A_main_e2e.md Q4/D6/D15/D17/D18, ledger ED-8, X-14):
  * refusal detector added (lexical by default, pattern list saved; LLM option);
  * output-guarded generations are no longer overwritten: the raw text, its judge verdict
    and the delivered refusal are all saved;
  * --benchmark-mode is explicit (default agnostic) and fixed before calibration;
  * temperature / top_p / top_k / seed / chat template / max_new_tokens are recorded, and
    the decoding description is derived from the settings instead of hard-coded;
  * the calibration protocol report (rcdp_guard.evaluation.fit_calibrator) is saved;
  * Python 3.8 compatible (no argparse.BooleanOptionalAction).
Old behaviour: --calibration-protocol legacy, --benchmark-mode aware,
--response-judge-backend api, --legacy-double-bos, --sampling-from-model-config,
--policy mtd-selection, --no-output-guard.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import e2e_common as E  # noqa: E402
from rcdp_guard.calibration import FeatureConfig  # noqa: E402
from rcdp_guard.calibration_protocol import PROTOCOLS  # noqa: E402
from rcdp_guard.controller import ControllerConfig, RCDPController  # noqa: E402
from rcdp_guard.datasets import load_examples  # noqa: E402
from rcdp_guard.evaluation import calibration_protocol_from_env, fit_calibrator  # noqa: E402

PROTOCOL_NOTES = [
    "State prompts are the submitted harness's texts (STATE_INSTRUCTIONS), not the shorter appendix templates; "
    "override with --state-prompts-file.",
    "Only Quarantine blocks before generation; Normal/Cautious/Hardened are sent to the protected model with the "
    "state instruction as system message and the last user turn as user message.",
    "The output check runs on generations routed to the states in configuration.output_guard_states (paper: "
    "Hardened) and replaces an 'unsafe' generation by the controller refusal; the raw text and verdict are kept.",
    "The primary / second judges score the raw generation; ASR counts only generations that were delivered "
    "(not replaced by the output check). asr_primary_before_output_check gives the pre-check count.",
    "Refusals are classified on the raw generation by the refusal detector in 'refusal_detector'.",
]


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    E.add_common_args(p)
    g = p.add_argument_group("controller")
    g.add_argument("--calibration-dataset", required=True)
    g.add_argument("--calibration-protocol", default=None, choices=list(PROTOCOLS),
                   help="Default: RCDP_CALIBRATION_PROTOCOL or r1 (paper protocol). legacy = submitted pipeline.")
    g.add_argument("--calibration-seed", default=None, help="Split seed (default RCDP_SPLIT_SEED or rcdp-r1).")
    g.add_argument("--benchmark-mode", choices=["agnostic", "aware"], default="agnostic",
                   help="agnostic = main configuration; aware = source-conditioned diagnostic (submitted Table 3).")
    g.add_argument("--benchmark-aware", dest="benchmark_mode", action="store_const", const="aware",
                   help="Alias for --benchmark-mode aware.")
    g.add_argument("--no-benchmark-aware", dest="benchmark_mode", action="store_const", const="agnostic",
                   help="Alias for --benchmark-mode agnostic.")
    g.add_argument("--num-states", type=int, default=4, choices=[2, 3, 4])
    g.add_argument("--policy", default="stateful-routing", choices=["stateful-routing", "mtd-selection"])
    g.add_argument("--scorer", default="distilbart-mnli", choices=["keyword", "distilbart-mnli", "llama-guard"])
    g.add_argument("--scorer-model-name", default="valhalla/distilbart-mnli-12-3")
    g.add_argument("--scorer-local-files-only", action="store_true")
    g.add_argument("--scorer-device", default="cpu")
    g.add_argument("--scorer-cache-path", default=None)
    g.add_argument("--scorer-cache-only", action="store_true")
    g.add_argument("--state-prompts-file", default=None,
                   help="JSON {normal, cautious, hardened} overriding the submitted state instructions.")
    g.add_argument("--keep-calibration-features", action="store_true")
    g.add_argument("--skip-generation", action="store_true",
                   help="Routing only (no protected model / judges): quick check of ABR_ctrl and BAR.")

    g = p.add_argument_group("Hardened output check")
    E.add_bool_flag(g, "output-guard", True, "check Hardened generations and replace unsafe ones by the "
                                             "controller refusal (paper: on)")
    g.add_argument("--output-guard-backend", choices=["downstream-judge", "llama-guard"], default="downstream-judge",
                   help="downstream-judge = reuse the primary judge verdict (submitted default).")
    g.add_argument("--output-guard-model-path", default=None)
    g.add_argument("--output-guard-device", default="cpu")
    g.add_argument("--output-guard-cache-path", default=None)
    g.add_argument("--output-guard-states", default="hardened",
                   help="Comma-separated states whose generations are checked, or 'all' (paper: hardened).")
    return p


def build_scorer(args):
    from rcdp_guard.scorers import DistilBartMNLIRiskScorer, KeywordRiskScorer, LlamaGuardRiskScorer
    if args.scorer == "keyword":
        return KeywordRiskScorer()
    if args.scorer == "llama-guard":
        return LlamaGuardRiskScorer(model_name=args.scorer_model_name, local_files_only=args.scorer_local_files_only,
                                    device=args.scorer_device, cache_path=args.scorer_cache_path)
    return DistilBartMNLIRiskScorer(model_name=args.scorer_model_name, local_files_only=args.scorer_local_files_only,
                                    device=args.scorer_device, cache_path=args.scorer_cache_path,
                                    cache_only=args.scorer_cache_only)


def state_prompts(args):
    prompts = dict(E.STATE_INSTRUCTIONS)
    if args.state_prompts_file:
        loaded = json.loads(Path(args.state_prompts_file).read_text(encoding="utf-8"))
        missing = {"normal", "cautious", "hardened"} - set(loaded)
        if missing:
            raise SystemExit("--state-prompts-file lacks %s" % sorted(missing))
        prompts.update({k: loaded[k] for k in ("normal", "cautious", "hardened")})
    return prompts


def main(argv=None, overrides=None):
    """``overrides`` (tests only): prebuilt scorer / generator / judge / second_judge /
    output_guard / refusal_detector objects that replace the CLI-built ones."""
    ov = dict(overrides or {})
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    log = ov.get("log", print)

    protocol = args.calibration_protocol or calibration_protocol_from_env()
    cal_examples = load_examples(args.calibration_dataset)
    examples = load_examples(args.dataset)
    E.check_partition(cal_examples, examples)
    if args.limit:
        examples = examples[: args.limit]
    E.warn_partition_size(examples, args.limit, log)

    # Benchmark mode must be set before fitting: fit_calibrator reads it (features and
    # per-benchmark thresholds).
    controller = RCDPController(
        scorer=ov.get("scorer") or build_scorer(args),
        feature_config=FeatureConfig.all(),
        config=ControllerConfig(num_states=args.num_states, benchmark_aware=(args.benchmark_mode == "aware")),
    )
    report = fit_calibrator(controller, cal_examples, protocol=protocol, seed=args.calibration_seed)
    log("[controller] protocol=%s benchmark_mode=%s thresholds=%s" % (protocol, args.benchmark_mode,
                                                                      controller.config.thresholds))

    prompts = state_prompts(args)
    generator = judge = second_judge = refusal = guard = None
    if not args.skip_generation:
        generator = ov["generator"] if "generator" in ov else E.build_protected_generator(args)
        judge = ov["judge"] if "judge" in ov else E.build_primary_judge(args, generator)
        second_judge = ov["second_judge"] if "second_judge" in ov else E.build_second_judge(args)
        refusal = ov["refusal_detector"] if "refusal_detector" in ov else E.build_refusal_detector(args)
        if args.output_guard:
            if "output_guard" in ov:
                guard = ov["output_guard"]
            elif args.output_guard_backend == "llama-guard":
                if not args.output_guard_model_path:
                    raise SystemExit("--output-guard-model-path is required for --output-guard-backend llama-guard")
                guard = E.LlamaGuardResponseJudge(args.output_guard_model_path, args.output_guard_device,
                                                  args.output_guard_cache_path, role="output-guard")
            else:
                guard = "reuse-primary"

    def prompt_builder(example, g, request):
        state = g["state"]
        return "state:" + state, [{"role": "system", "content": prompts[state]}, {"role": "user", "content": request}]

    guard_states = "all" if args.output_guard_states == "all" else tuple(
        s.strip() for s in args.output_guard_states.split(",") if s.strip())
    pipeline = E.Pipeline(
        gate=E.RCDPGate(controller), generator=generator, judge=judge, prompt_builder=prompt_builder,
        second_judge=second_judge, output_guard=guard, output_guard_states=guard_states,
        refusal_detector=refusal, generation_seed=args.generation_seed, skip_generation=args.skip_generation,
        policy="mtd-selection" if args.policy == "mtd-selection" else "gate",
    )

    configuration = {
        "harness": "rcdp",
        "policy": args.policy,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_aware": args.benchmark_mode == "aware",
        "num_states": args.num_states,
        "calibration_protocol": protocol,
        "calibration_seed": args.calibration_seed,
        "controller_scorer": args.scorer,
        "controller_scorer_model": args.scorer_model_name if args.scorer != "keyword" else None,
        "controller_scorer_cache_path": args.scorer_cache_path,
        "output_guard": bool(args.output_guard),
        "output_guard_backend": args.output_guard_backend if args.output_guard else None,
        "output_guard_model_path": args.output_guard_model_path if args.output_guard else None,
        "output_guard_states": guard_states if args.output_guard else None,
        "skip_generation": args.skip_generation,
        "generation_seed": args.generation_seed,
        "limit": args.limit,
    }
    gen_settings = E.describe(generator)
    fingerprint = E.config_fingerprint({"argv": [t for t in E.redacted_argv(argv) if t != "--resume"],
                                        "configuration": configuration,
                                        "generation": gen_settings, "dataset": args.dataset})
    partial = Path(args.output + ".partial.jsonl")
    records = E.run_items(examples, pipeline, partial, args.resume, fingerprint, log)

    result = {
        "schema": E.SCHEMA,
        "run": E.run_info(argv, [__file__, E.__file__]),
        "configuration": configuration,
        "datasets": {"evaluation": E.dataset_info(args.dataset, examples),
                     "calibration": E.dataset_info(args.calibration_dataset, cal_examples)},
        "generation_settings": gen_settings,
        "system_prompts": {"state:" + k: v for k, v in prompts.items()},
        "controller_refusal_text": E.CONTROLLER_REFUSAL,
        "gate": pipeline.gate.describe(),
        "judges": {"primary": E.describe(judge), "secondary": E.describe(second_judge),
                   "output_guard": E.describe(guard)},
        "refusal_detector": E.describe(refusal),
        "metric_definitions": E.METRIC_DEFINITIONS,
        "protocol_notes": PROTOCOL_NOTES,
        "calibration_report": E.slim_calibration_report(report, args.keep_calibration_features),
        "summary": E.summarize(records, 0 if args.skip_generation else args.bootstrap_samples),
        "legacy_summary": E.legacy_summary(records),
        "records": records,
    }
    E.write_result(args.output, result, partial)
    log(json.dumps({"output": args.output, "configuration": configuration,
                    "headline": result["summary"]["headline"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
