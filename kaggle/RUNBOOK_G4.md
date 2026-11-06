# RUNBOOK G4: full LoRA fine-tuning run on Kaggle (for Thông)

Goal: train the FraudLM LoRA adapter on GPT-2 (size chosen by G2) on a free Kaggle GPU as a background
**commit** ("Save & Run All"). The run keeps going after you close the browser (up to 12 h). You upload
the dataset yourself through the Kaggle website. No agent uploads anything, and no token goes into any file.

Before you start, the lead has:
- filled the G2 values (size, base_dtype, grad_ckpt, micro_batch, grad_accum) into `DEFAULTS` in
  `training_model/train_sft.py`,
- run `python kaggle/make_bundle.py`, which writes `kaggle/train_bundle.py` (header says "GENERATED ... on
  <date time>") and puts the dataset fingerprint into it, and
- run `pytest tests/test_train_smoke.py`. Its bundle-is-current tests must pass.
- told you the **run id**. It is derived from the size: `xl_r16_v1` for GPT-2 XL, `large_r16_v1` for Large,
  `medium_r16_v1` for Medium. Below, `<run_id>` means that name.

## 1. Upload the dataset (once per dataset version, about 2 min)

1. Go to https://www.kaggle.com/datasets and click **+ New Dataset**.
2. Drag in these 4 files from `D:\School Courses\Sem4\Concept Innovative\assignment-2\llm_ver2\dataset\`:
   `fraudlm_sft_train.json` (about 10 MB), `fraudlm_sft_val.json`, `fraudlm_sft_test.json`,
   `fraudlm_demo_months67.json`. Only train and val are used for training. Test and the months 6-7 demo
   file are there for the G5 evaluation (`kaggle/RUNBOOK_G5.md`).
   (Done by the lead on 06/11: private dataset `chunghinthng/fraudlm-sft-v1-1`, all 4 files.)
3. Title: `fraudlm-sft-v1-1`. Visibility: **Private**. Click **Create**. Wait until it shows "processed".
4. If the data is rebuilt later, open the dataset and use **New Version** with the new files. Don't create a
   second dataset, because then the bundle finds two copies. The bundle checks the files' fingerprint
   and stops at once if the Kaggle copy is not the one it was built with.

## 2. Create the training notebook

1. https://www.kaggle.com/code → **+ New Notebook**.
2. Right panel → **Session options**: **Accelerator = GPU T4 x2** (or the GPU that G2 used). **Internet = On**
   (needed to download the GPT-2 weights; the setting is saved with the notebook and applies to the commit).
3. Right panel → **+ Add Input** → **Your Work** (or Datasets → "Your Datasets") → `fraudlm-sft-v1-1` → **Add**.
   It appears under `/kaggle/input/`. The bundle searches there by itself.
4. Rename the notebook (top left) to `fraudlm-train`.
5. Open `llm_ver2\kaggle\train_bundle.py` on your PC, **Ctrl+A**, **Ctrl+C**. In the notebook's first cell,
   delete what is there and press **Ctrl+V**. Leave the code as it is: every setting is already in the bundle.
   **Do not run the cell here.** That would start the 3-5 h run in the interactive session.

## 3. Start the run as a background commit

1. Top right: **Save Version** → choose **Save & Run All (Commit)** → **Save**.
2. Stop the interactive editor session (power button, top right). The commit runs in its own session, so
   the editor session would only use up GPU quota.
3. **First 10 minutes check.** Open the notebook → **Version history** (or the bell / "Active events") →
   click the running version → **Logs**. Before you leave, check that the log shows all of these, in order:

   | Look for | Must be |
   |---|---|
   | `[gpu] ...` | `Tesla T4` (or the GPU G2 used), about 14.7 GB |
   | `[data] ...: sha ... = bundle (OK)` | the words `= bundle (OK)` |
   | `[data] train: 12,000 examples, ... cap 512` | 12,000 examples (or the number the lead gives you) and **no** `WARNING` on that line |
   | `[plan] ... steps total` | 3,000 steps (or the value the lead gives you) |
   | `[eval] step 0: val subset loss X` | X is a normal number, about 2-3 (not `nan`, not above 5) |
   | first `step 10/...` line | `s/step` within about 20% of the G2 number the lead gives you |

   If any of these is wrong, click **Cancel** on the running version and send the log to the lead.
   Each later log line comes every 10 steps:
   `step 120/3,000 | ep 0.08 | loss 1.2345 | lr 9.9e-05 | 4.10 s/step | elapsed 0:09:12 | ETA 3:15:00 | scale 65536, skipped 2`.
   Every 200 steps it adds a val loss and saves a checkpoint (`[save] step 200 -> ...`).
   An `[amp] WARNING` line means the run is barely learning. Send the log to the lead.
4. Now you can close the browser. The run is finished when the log ends with
   `[done] 3,000 steps in ...; full val loss (500 rows) ...`.

## 4. Download the result

1. Open the finished version → **Output** tab → folder `checkpoint/<run_id>/` → **Download** (or "Download all").
2. Unzip it into `D:\School Courses\Sem4\Concept Innovative\assignment-2\llm_ver2\checkpoint\<run_id>\`.
   You should have: `lora_adapter.pt` (final adapter), `lora_adapter_best.pt` (lowest val loss),
   `train_state.pt` (resume state with the adapter and optimiser, about 3× the adapter size), `run_config.json`,
   `train_log.csv`, `samples.txt`, `progress.json`, `run_summary.json`. The GPT-2 base weights are **not** in
   the output (they stay in `/tmp`), so the download is a few hundred MB at most.
3. Tell the lead "G4 run done" and paste the last 20 lines of the log.

## 5. If the run stops early

| What you see | What to do |
|---|---|
| Log ends with `[time] --max_hours 11.0 reached at step N ... Saved.` (version status "complete") | Resume: open the notebook (Edit) → **+ Add Input** → **Your Work** → **Notebooks** → `fraudlm-train` (the version that just ran) → Add. Then **Save Version → Save & Run All** again. The bundle finds `checkpoint/<run_id>/train_state.pt` under `/kaggle/input/`, copies it to `/kaggle/working/` and continues from step N. **Check the log:** it must say `[resume] from step N / ...`. If it shows `[eval] step 0` instead, the previous output was not found: **Cancel** at once, or the run starts again from step 0 and spends the hours again. Download from the **newest** version. |
| Version **failed** or was stopped by Kaggle in the middle (GPU error, the 12 h limit, a Kaggle restart) | It is not known whether a failed version keeps its output. The `--max_hours` guard (11 h) is there so the run ends as a clean "complete" version first. If the failed version still has an Output tab with `checkpoint/<run_id>/train_state.pt`, resume as in the row above. Otherwise start a new version without an input notebook: the run restarts from step 0. |
| `[data] the dataset found in ... differs from the one this bundle was built with` | The Kaggle dataset is out of date or the bundle is old. Upload the current `dataset\` files as a **New Version** of `fraudlm-sft-v1-1` (step 1.4), or ask the lead to rebuild the bundle. |
| `[eval] step-0 val loss ... is not sane` | The base model did not load correctly in fp16. Send the log to the lead (the fix is base_dtype fp32). Nothing was trained. |
| `[nan] STOPPED: ... non-finite losses` | The version still ends as "complete" and keeps the last good checkpoint. Send the last 40 log lines to the lead (the fix is base_dtype fp32 or a lower lr). |
| `[data] fraudlm_sft_train.json not found` | Step 2.3 is missing: add the dataset as an input. |
| `ConnectionError` / `Max retries exceeded` during `[weights] downloading` | Internet is Off (step 2.2) or your phone is not verified. |
| `[run] ... already holds a checkpoint` | Only happens with `resume` switched off. Tell the lead. |
| `CUDA out of memory` | Send the last 40 lines to the lead (the fix is a smaller micro_batch with a larger grad_accum). |

Notes
- Quota: one full run uses about the projected hours from G2 (plus ~10 min for download and setup) of your
  30 h weekly GPU quota. A stopped editor session uses no quota.
- The run is deterministic (seed 42). A resumed run continues exactly as an uninterrupted one would
  (tested on CPU in `tests/test_train_smoke.py`).
- Never paste a Kaggle or Hugging Face token anywhere. The weights are public and the dataset is your own private upload.
