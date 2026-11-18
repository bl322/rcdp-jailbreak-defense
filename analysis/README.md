# Analysis scripts

Every number in the tables, figures and text of the revised manuscript was computed from the per-example result
records by the scripts in this directory. They are released as they were run, with their working-directory layout
(`ccfa-workfiles/tables/...`, `_r1_out/...`). `AS_RUN.sha256` gives their hashes as run. The only change is in the
four scripts that check the SHA-256 of an input result file: on the redacted package they use the hash of the
original file from `MANIFEST.json` (`as_run.diff`).

## Running them on the released records

The records are the asset `rcdp_r1_results_redacted.tgz` of the release. `setup_layout.py` places every result file
under its original path and the scripts beside it:

```bash
tar xzf rcdp_r1_results_redacted.tgz                    # -> rcdp_r1_results_redacted/
python analysis/setup_layout.py rcdp_r1_results_redacted work
python work/ccfa-workfiles/tables/single-turn/source/single_turn_stats.py
# output: work/ccfa-workfiles/tables/single-turn/build/
```

Run the scripts of a group in the order listed; each writes to the `build/` directory next to its `source/`.
Output file names keep the numbering used while the revision was prepared (for example `tableS12_rows.tex` is
Supplementary Table S9).

| script (under `work/ccfa-workfiles/`) | manuscript |
|---|---|
| `tables/single-turn/source/collect_single_turn.py`, `single_turn_stats.py`, `make_single_turn_tables.py` | Tables 1 and 2; Section 4.2 (paired McNemar tests, bootstrap intervals, AUC, ECE, Brier); Supplementary Tables S10a, S10b, S11, S12, S13 |
| `tables/single-turn/source/vif_head.py` | appendix collinearity table |
| `tables/single-turn/source/verify_baselines.py` (arguments: the `e2e_qwen` baseline and JailGuard files) | Table 1(a), SmoothLLM-style and JailGuard-style rows |
| `tables/single-turn/source/verify_sensitivity.py` | Supplementary Section S12, split seeds |
| `tables/splits/source/verify_splits.py <work dir>` | Supplementary Table S1 (also rewrites `data/paper_splits/split_ids_r1.json`) |
| `tables/e2e/source/collect_e2e.py` | Table 3; Supplementary Tables S7 and S14 |
| `tables/e2e/source/collect_conditions_e2e.py` | Sections 4.2.3 and 4.2.4 (generation conditions, 346-example pool) |
| `tables/e2e/source/state_layer.py`, `state_layer_hardened.py` | Supplementary Table S8; appendix output-check counts |
| `tables/e2e/source/verify_second_judge.py`, `state_layer_second_judge.py`, `table_s12_judges.py` | Section 4.2.4 (second judge); Supplementary Table S9 |
| `tables/e2e/source/verify_orbench.py` | Sections 4.1 and 4.2.2, Supplementary Section S11 (OR-Bench) |
| `tables/adaptive/source/verify_tap.py`, `verify_tap_lg.py`, `verify_r5b.py --root <work dir>` | Table 4; Section 4.3.1; Supplementary Tables S6a and S6b |
| `tables/adaptive/source/verify_r5b_cost.py --root <work dir>` | Section 4.3.1; Supplementary Table S6c |
| `tables/cost/source/verify_cost.py` | Section 4.4; appendix latency table |
| `figures/fig4-multiturn-frontier/source/make_tables.py`, `build_fig4.py` | Figure 4; Supplementary Tables S5a, S5b, S5c |
| `figures/figS2-reliability/source/build_figS2.py` | Supplementary Figure S1 |
| `scripts/fixed_fpr_r1.py`, `scripts/plot_cost_r1.py` (repository root; inputs in `PROVENANCE.md`) | Figure 3 and Supplementary Table S3; Figure 5 |

Supplementary Table S5d lists the candidate set and the selected configurations recorded in the `policy_search`
block of the multi-turn result files.

## What matches byte for byte

On the redacted package every script above reproduces its output files as run, byte for byte, except:

- `verify_orbench.py` also reads the OR-Bench-hard prompts, which the package does not carry. Build them with
  `python scripts/prepare_data.py` and copy `data/external/orbench_hard_1k.jsonl` and
  `data/external/orbench_hard_sub500.jsonl` into `work/_r1_out/r4/extracted/r4_pack/`; its outputs are then identical as well.
- PDF figures embed their creation date (their PNG previews and CSV files are identical).
- The JSON records of `fixed_fpr_r1.py` and `plot_cost_r1.py` differ only in the recorded input paths and
  input-file hashes.
