# RUNBOOK G5: evaluation of B0, FT and B1 on Kaggle

Goal: score the fine-tuned model (FT = GPT-2 XL + the LoRA adapter of run `xl_r16_v1`) against B1 (the same GPT-2 XL,
no adapter) and B0 (the rule responder) on the test split (500 rows) and the months 6-7 demo file (300 rows),
using the evaluator's harness (`evaluation/run_eval.py`, metrics in `evaluation/EVAL_NOTES.md`).
The bundle is `kaggle/eval_bundle.py`, built from `kaggle/eval_job.py` + the repo modules incl. `evaluation/`.
No token goes into any file. Agents never push. The lead (authorised by Thông) or Thông starts the run.

## 0. Before the run

- The training kernel `chunghinthng/fraudlm-train` has finished, and its log ends with `[done] ...`. Its output holds
  `checkpoint/xl_r16_v1/lora_adapter.pt`, `run_config.json` and `train_log.csv`.
- Rebuild only the eval bundle if `evaluation/` or `kaggle/eval_job.py` changed:
  `.venv/Scripts/python kaggle/make_bundle.py --only eval`. make_bundle keeps a bundle file byte for byte when
  only its time stamp would change, so `train_bundle.py` (the code of the running training job) stays identical.
- `pytest tests/test_eval_job.py -q` must pass (includes "eval bundle is current").

## 1. Route A: script kernel pushed by the lead (CLI)

Kernel folder, e.g. `kaggle/kernels/eval/`:
- `fraudlm_eval.py` = an argv line, then the whole of `kaggle/eval_bundle.py`:
  ```python
  import sys
  sys.argv = [sys.argv[0], "--max_hours", "11"]
  ```
  Nothing else is needed. The adapter and the data are found under `/kaggle/input/**`. The base precision is
  taken from the run's `run_config.json` (`base_dtype` fp32 for `xl_r16_v1`, the same as training; the
  adapter stays fp32 and unmerged). Useful extra flags:
  - `--models ft,b1` / `--splits test`: run part of the grid (default `b0,ft,b1` × `test,demo67`, in that order:
    B0 takes seconds on the CPU, then FT, then B1).
  - `--limit 20`: a seeded 20-row subset per split (smoke test). Its file names end in `-n20`, so it never
    overwrites a full run.
  - `--adapter /kaggle/input/fraudlm-train/checkpoint/xl_r16_v1/lora_adapter_best.pt`: score the lowest
    val-subset-loss adapter instead of the final one (default `lora_adapter.pt`). Result files are named by the run folder only,
    so run this in a separate kernel with no earlier eval output attached (otherwise finished files count as done).
  - `--run_id xl_r16_v1`: only needed if adapters of several runs are attached.
- `kernel-metadata.json` (same shape as the training kernel):
  ```json
  {"id": "chunghinthng/fraudlm-eval", "title": "fraudlm-eval", "code_file": "fraudlm_eval.py",
   "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
   "enable_tpu": "false", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
   "dataset_sources": ["chunghinthng/fraudlm-sft-v1-1"], "competition_sources": [],
   "kernel_sources": ["chunghinthng/fraudlm-train"], "model_sources": []}
  ```
  Internet must be on: the GPT-2 XL weights are downloaded from the Hugging Face hub (public, no token) to `/tmp`.

## 2. Route B: Thông in the web UI (if the lead is not available)

1. https://www.kaggle.com/code → **+ New Notebook** → Session options: **GPU T4 x2**, **Internet On**.
2. **+ Add Input** → Datasets → Your Datasets → `fraudlm-sft-v1-1`, and **+ Add Input** → Your Work →
   Notebooks → `fraudlm-train` (its latest finished version).
3. Paste `kaggle/eval_bundle.py` into one cell. Do not run it interactively.
4. **Save Version → Save & Run All (Commit)**, then stop the editor session.

## 3. What the log must show (first ~10 minutes)

| Look for | Must be |
|---|---|
| `[eval_job] run xl_r16_v1: gpt2-xl, adapter /kaggle/input/fraudlm-train/checkpoint/xl_r16_v1/lora_adapter.pt, base_dtype fp32` | run id, size and base dtype as trained |
| `[eval_job] data /kaggle/input/fraudlm-sft-v1-1` | the dataset is found (no `demo67 skipped` warning) |
| `=== B0 on test` / `=== B0 on demo67` | finish in seconds, with a short metrics table |
| `[weights] downloading openai-community/gpt2-xl ...` | once (1-5 min). Later loads use the `/tmp` cache |
| `[eval_job] first rows: X s/row` (rows 1 and 3), then `[eval] 10/500 rows, X s/row, ETA Y min` every 10 rows | X about 1.5-4 s/row on a T4 (estimate, see 5). The ETA is per split |
| `[eval] response loss pass: ... s` then `[eval] done: .../xl_r16_v1_FT-gpt2-xl_test.json` | per split |
| last line `[eval_job] DONE: 6 result files ...` | status also in `evaluation/eval_status.json` |

## 4. Outputs (in the kernel output, `/kaggle/working/evaluation/`)

- `results/xl_r16_v1_{B0,FT-gpt2-xl,B1-gpt2-xl}_{test,demo67}.json` (+ `_rows.csv`): summary and every row.
  `results/summary.csv` is the long table, and `results/*_FT-*_examples.md` holds the 3 good and 3 bad examples (fixed rules).
- `figures/loss_curve_xl_r16_v1.png` (from the training run's `train_log.csv`) and
  `figures/metrics_bars_xl_r16_v1_{test,demo67}_all.png`, each with a `.json` that lists the plotted numbers.
- `eval_status.json`: `done` or `time_limit`, plus the finished files.

Download the `evaluation/` folder into `llm_ver2/evaluation/kaggle_xl_r16_v1/`. Then the evaluator or lead
copies the results into `evaluation/results/`. Quote numbers only from these files (EVAL_NOTES rules).

## 5. Time (ESTIMATE, not measured on XL yet)

- Answers are short: the reference responses average 54 tokens (test 54.4, demo 54.8; one test row is above 200).
  B1 on gpt2-small averaged 26-41 new tokens before the `###` stop. Generation is greedy with the KV cache, batch 1, cap 200.
- Assumed XL fp32 decode speed on a T4: about 40-60 ms per token (6.2 GB of weights read per token, plus
  per-layer kernel overhead), plus about 0.2 s of prompt prefill per row.
- FT: 800 rows × ~55 tokens → about 35-50 min. If FT often runs to the 200-token cap: up to about 2.5 h.
- B1: 800 rows × ~35 tokens → about 25-35 min. B0: seconds. Four XL loads + response-loss passes: about 10 min.
- **Total estimate about 1-1.5 h (worst case about 3.5 h)**, well inside `--max_hours 11`. Check the measured
  `first rows: X s/row` line to replace this estimate.

## 6. If it stops early

- `[eval_job] --max_hours ... reached during ...` (status `time_limit`, version "complete"): every finished row is
  in `results/*.partial.jsonl`. Push or run a new version with **this eval kernel's own output added as an
  input** (Route A: add `"chunghinthng/fraudlm-eval"` to `kernel_sources`; Route B: + Add Input → Your Work →
  fraudlm-eval). The job copies those files back (`[eval_job] restored N result files`), skips finished splits
  (`... already done, skipped`) and resumes the unfinished one (`[eval] resume: N rows already generated`).
- The script always exits 0 on Kaggle, so a time-limit stop still ends as a clean "complete" version with its
  output. It is not known whether a version killed by Kaggle (12 h limit, GPU error) keeps `/kaggle/working`.
- `no checkpoint/<run_id>/lora_adapter.pt with run_config.json found`: the training output is not attached (step 1
  `kernel_sources`, or Route B step 2).
- `adapters of several runs found`: pass `--run_id xl_r16_v1`.
