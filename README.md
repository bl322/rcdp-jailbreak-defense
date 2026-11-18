# RCDP: Risk-Calibrated Dynamic Prompting

Code for *"Risk-Calibrated Stateful Control for Inference-Time Jailbreak Defense"* (Computers, Materials &
Continua, manuscript 95311, first revision). RCDP scores each request, calibrates the score into a risk estimate,
and routes the conversation through a four-state controller (Normal / Cautious / Hardened / Quarantine).

```
rcdp_guard/     the method: calibration protocol, controller, scorers, multi-turn policies
scripts/        experiment scripts (index in scripts/README.md)
reproduce/      one script per group of tables and figures, with the commands that produced them
data/           data preparation output (built by scripts/prepare_data.py); split identifiers
analysis/       scripts that computed the reported numbers from the result records (analysis/README.md)
PROVENANCE.md   for every result file: the commit whose code produced it and the recorded command
```

## Installation

Python 3.10 or later, PyTorch with CUDA for the GPU experiments.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt matplotlib huggingface_hub
```

## Models

```bash
huggingface-cli login      # Llama-Guard-3-1B and Llama-3.1-8B-Instruct are gated; accept their licences first
huggingface-cli download valhalla/distilbart-mnli-12-3                 # used at revision ef9a58ce
huggingface-cli download MoritzLaurer/deberta-v3-large-zeroshot-v2.0   # used at revision cf44676c
huggingface-cli download meta-llama/Llama-Guard-3-1B --local-dir llama-guard-3-1b
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Meta-Llama-3.1-8B-Instruct
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/Mistral-7B-Instruct-v0.3
```

The two NLI scorers are loaded from the Hugging Face cache; on 2026-11-18 the default revision of both
repositories was the one used. Llama-Guard must sit in `llama-guard-3-1b/` at the repository root (the TAP and
policy scripts look for it there). Other locations can be set in `reproduce/env.sh`.

The end-to-end, TAP and second-judge experiments also call an OpenAI-compatible endpoint (`RCDP_API_BASE`,
`RCDP_API_KEY`) that serves `qwen3-next-80b-a3b-instruct` (the hosted protected model, the second judge, and the
TAP attacker and target judge) and `deepseek-v4-flash-0731` (the TAP re-judge); the names are those of the
endpoint used in the experiments.

## Data

```bash
python scripts/prepare_data.py --download
```

This downloads the public sources (JailbreakBench, HarmBench, XSTest, OR-Bench, SafeMTData/ActorAttack, MS-XSB,
MT-Bench-101) into `data/raw/`, builds every data file the experiments read, and compares each with the file the
experiments used (SHA-256). It takes about two minutes and prints one line per file; files with a recorded hash
must be identical. The split rules are in the docstring of `scripts/prepare_data.py`;
`data/paper_splits/split_ids_r1.json` lists the identifiers of the calibration subsets.

MHJ (ScaleAI/mhj) is gated and is not redistributed. With access, save its `harmbench_behaviors.csv` as
`data/raw/MHJ_harmbench_behaviors.csv` and rerun the command; without it, the MHJ runs are skipped.

## Reproducing the experiments

| manuscript | command | runs | needs |
|---|---|---|---|
| Tables 1 and 2, Fig. 3; Supp. Tables S1, S3, S10–S13 | `bash reproduce/01_single_turn.sh` | 14 | one GPU |
| Fig. 4; Supp. Tables S5a–S5d | `bash reproduce/02_multi_turn.sh` | 6 | one GPU |
| Table 3; Supp. Tables S7, S8, S9, S14 | `bash reproduce/03_end_to_end.sh` | 28 | GPU, endpoint, local LLMs |
| Sections 4.2.3–4.2.4 (generation conditions, 346-example pool) | `bash reproduce/04_generation_conditions.sh` | 24 | GPU, endpoint, local LLMs |
| Sections 4.1, 4.2.2; Supp. Section S11 (OR-Bench) | `bash reproduce/05_orbench.sh` | 9 | GPU, endpoint |
| Table 4; Supp. Tables S6a, S6b (adaptive TAP) | `bash reproduce/06_adaptive_tap.sh` | 16 | GPU, endpoint; after 03 |
| Section 4.3.1; Supp. Table S6c (per-caller policies) | `bash reproduce/07_policy_cost.sh` | 2 | GPU; after 03 |
| Fig. 5; Section 4.4; appendix latency tables | `bash reproduce/08_serving_cost.sh` | 6 | 16-core CPU and GPU |

Each command in these scripts is the one recorded by a result file of the paper, with model paths replaced by the
variables of `reproduce/env.sh`; the comment above it names that result file in the released package and the
commit that produced it. Steps whose output does not store its command (refusal re-check, second judge, TAP
re-judging, matched bootstrap) use the settings their output files record. The scripts write to `results/`.

The scripts run on the code of this release, which accepts every recorded command. Some experiments ran on
earlier commits; to run the code exactly as it ran, check out the commit that `PROVENANCE.md` lists for the result
file. Scorer outputs on a different GPU can differ in the last digits, and generations of a hosted endpoint depend
on the server.

## Checking the reported numbers

Release `v1.1.0` carries the asset `rcdp_r1_results_redacted.tgz`: every result file of the revision, with
identifiers, labels, scores, controller states and decisions, judge verdicts, settings and code hashes. Prompt,
dialogue and generated text, attacker feedback and judge outputs are removed (`<redacted:N chars>`). Every number
of the paper can be recomputed from it without a GPU:

```bash
tar xzf rcdp_r1_results_redacted.tgz
python analysis/setup_layout.py rcdp_r1_results_redacted work
python work/ccfa-workfiles/tables/single-turn/source/single_turn_stats.py
```

`analysis/README.md` lists the script behind each table and figure. To compare a rerun, compare its file in
`results/` with the released file of the same name.

## Code versions

The experiments of the revision ran while the code was being extended. Each result file records the SHA-256 of the
code it ran, and every code state that produced a reported result is a tagged commit:

| tag | code |
|---|---|
| (commit `22b72dd`) | submitted version, October 2026 (its outputs are in that commit's `results/`) |
| `r1-stage1` … `r1-server-r5-lgfix` | code states of the revision experiments, in deployment order |
| `v1.1.0` | this release |

`PROVENANCE.md` maps each result file to its commit; `python scripts/verify_provenance.py <unpacked package>/results
--repo .` repeats the check.

## Calibration protocol

`rcdp_guard/calibration_protocol.py` implements the protocol of the paper; `rcdp_guard.evaluation.fit_calibrator`
uses it by default.

1. The calibration pool is split into disjoint subsets D_head / D_platt / D_thresh (100 / 48 / 48 of 196),
   ordered by (label, benchmark, SHA-256(`"<seed>:<example_id>"`)); the seed defaults to `rcdp-r1`.
2. An L2 logistic risk head (C = 0.5, learning rate 0.1, 300 full-batch epochs) is fitted on D_head.
3. Platt scaling r = σ(a·s̃ + b_cal) is fitted on the head logits of D_platt.
4. The blocking threshold is chosen on D_thresh by maximising J = BlockRate − 1.0·FPR over τ ∈ {0.01, …, 0.99}.
   Ties go to the higher BlockRate, then to the median of the tied run. τ_c and τ_h keep their default ratios to
   τ_q. In benchmark-aware mode one τ_q is chosen per benchmark.

The submitted runs used an unpenalised head on the whole pool with default thresholds and a turn-1 trend boost
that blocked single-turn requests at r ≥ 0.6911 instead of τ_q. `RCDP_CALIBRATION_PROTOCOL=legacy`
(`--variants legacy-submitted` in `run_r1_main.py`) reproduces those runs exactly.

| variable | default | meaning |
|---|---|---|
| `RCDP_EXPERIMENTS` | repository root | directory that holds `rcdp_guard/`, `data/` and `results/` |
| `RCDP_API_BASE` | `http://localhost:30001/v1` | OpenAI-compatible endpoint |
| `RCDP_API_KEY` / `OPENAI_API_KEY` | none | key for that endpoint |
| `RCDP_CALIBRATION_PROTOCOL` | `r1` | `r1`, `legacy` (submitted pipeline), `r1-no-platt`, `raw-threshold` |
| `RCDP_SPLIT_SEED` | `rcdp-r1` | seed of the three-way split |

## Notes

All attack prompts come from public benchmarks and are used only to evaluate defenses. Licence: MIT (`LICENSE`).
