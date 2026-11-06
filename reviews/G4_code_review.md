# G4 code review (pre-run): training script, data loader, Kaggle bundle, runbook

Reviewer agent, 2026-11-06 06:20 (UTC+7). Scope: `training_model/sft_data.py`, `training_model/train_sft.py`,
`kaggle/make_bundle.py`, `kaggle/train_bundle.py` (generated 06:01), `kaggle/RUNBOOK_G4.md`,
`tests/test_sft_data.py`, `tests/test_train_smoke.py`. The full XL run on Kaggle has **not** happened, and G2 GPU
numbers are still pending, so the "val loss falls; run finished" half of G4 is not assessed here.

## Verdict: **PASS WITH FIXES**: fix R1 and R2 before the Kaggle launch; R3 and R8 are strongly advised

The masked loss, the fp16/LoRA dtype recipe, gradient checkpointing, the batch order on resume and the bundle are
correct. One definite crash: **every resume on the GPU fails** (R1). The only thing that lets the run outlive one
Kaggle session (time guard, then resume) is broken there, but the first session itself is not affected. R2 and R3
are cheap guards against a silent mis-train (stale data, broken fp16 base) that would otherwise only show up hours
later. Each is a few lines. After the fixes the trainer re-runs `make_bundle.py` and `pytest tests -q`; no full
re-review is needed, and the reviewer re-checks only the changed lines.

Test run by the reviewer: `.venv/Scripts/python -m pytest tests -q` → **96 passed in 74.79s** (includes
`test_committed_train_bundle_is_current`, so the bundle matches the repo).

## Checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Label shift done once | PASS | `sft_data.py:69-77`: inputs = seq[:-1], targets = seq[1:], first `len(prompt)-1` targets → -100. `train_sft.py:275` compares logits[t] with y[t] and does no second shift. |
| 1 | Masked loss = mean over response tokens (independent re-derivation) | PASS | Reviewer script: for 3 val rows, tokenised `prompt+response` jointly and summed `-log p(token j | <j)` for j ≥ len(prompt). Result 2.815044 vs `batch_loss` 2.815043 (diff 5.5e-7), 273 learned tokens on both sides. EOS is the last learned target (asserted `full[-1]==50256`). |
| 1 | Padding | PASS | `collate` right-pads with x=EOS and y=-100 (`sft_data.py:142-149`). Causal SDPA (`gpt_model.py:64-68`) means real tokens never see the pads after them. Reviewer: padded and unpadded logits on real tokens differ by ≤1.1e-4 (pads 129 and 145; kernel rounding only). |
| 1 | Gradient accumulation | PASS, note R6 | `train_sft.py:496` uses `loss/len(group)`, so the result is a mean of micro-batch means. The last group of an epoch is shorter and is handled by `len(group)`. |
| 2 | Base fp16 before apply_lora; LoRA fp32; optimiser on LoRA only | PASS | `train_sft.py:370-383`. Assert on line 379. `params` = requires_grad only. No `.half()`/`.to(dtype)` after LoRA (grep). |
| 2 | autocast fp16 + GradScaler, `unscale_` before clip, loss on `logits.float()` | PASS | `train_sft.py:272-275, 385, 496, 513-517`. Reviewer replayed the exact GPU recipe on CPU (fp16 base, fp16 autocast, **enabled** GradScaler, ckpt): see "fp16 simulation" below. |
| 2 | fp16 overflow risk in our model | MEDIUM (silent), see R3/R8 | Analysis below. gpt2-small fp32 measured: max abs residual 3000.2, max LayerNorm variance 12,480 (fp16 max 65,504; 5× headroom), max abs GELU input 62.8, max abs logit 341.6. XL could not be measured here. |
| 3 | Grad checkpointing `use_reentrant=False` + autocast + frozen base | PASS | `gpt_model.py:192-194`. Reviewer: bf16 autocast on CPU gives identical LoRA grads with and without ckpt (max diff 0.0), all fp32 and finite. Non-reentrant ckpt restores the autocast state on recompute and handles a block input that needs no grad. drop_rate=0, so RNG replay does not matter. |
| 4 | Resume: batch order, step, LR, scaler, optimiser | PASS on CPU | Batches are rebuilt from `seed+epoch` with `batch_pos` (`train_sft.py:466, 478-481`); LR comes from the step (`lr_at`). Kill/resume test passes (`test_resume_after_kill_continues_identically`). |
| 4 | Resume on GPU | **FAIL → R1** | `train_sft.py:401` loads with `map_location=device` (cuda), which moves the saved RNG ByteTensors to the GPU. `torch.set_rng_state` (line 406) and `torch.cuda.set_rng_state` (line 409) accept only CPU ByteTensors. Reviewer: `torch.set_rng_state(state.to("meta"))` raises `TypeError: RNG state must be a torch.ByteTensor`, and `map_location` moves the ByteTensor (meta device seen after load). The CPU tests cannot catch this. |
| 4 | Auto-resume from `/kaggle/input` | PASS | `find_resume_dir` (`train_sft.py:224-249`) picks the copy of `<run_id>/` with the highest `progress.json` step, so the folder name does not matter. It is copied to `/kaggle/working` before loading (line 397-400). |
| 4 | Locked settings | PASS | `RESUME_LOCKED` covers size, LoRA, dtype, batch/accum, max_len, epochs, LR schedule, seed, limits, data sha. A mismatch raises SystemExit before any GPU time is spent. |
| 4 | Atomic saves | PASS per file, note R5 | `os.replace` of a `.tmp` (`train_sft.py:211-221`). The adapter and `train_state.pt` are two separate files (R5). |
| 4 | `--max_hours` guard | PASS | `session_start` is set at the top of `train()` (line 331), before the data load and the 6.4 GB weight download. Not counted: bundle Part A/B (seconds) and container boot. The check runs before every step (line 472). Worst overrun after 11 h is one step plus one eval (minutes), against 60 min of margin. |
| 5 | Kaggle paths | PASS | out `/kaggle/working/checkpoint`, cache `/tmp/fraudlm_pretrained` (the base is not in the output), modules in `/tmp/fraudlm_src`. |
| 5 | Notebook argv / exit | PASS | Reviewer: `parse_config(["-f", ".../kernel-abc.json"])` parses (parse_known_args). No `sys.exit` under ipykernel (line 583). Config errors raise SystemExit, so the cell fails loudly before training, which is what we want. |
| 5 | Output size | PASS | XL r16 attn+ff = 22,118,400 fp32 params: `lora_adapter.pt` 88.5 MB, `lora_adapter_best.pt` 88.5 MB, `train_state.pt` ≈177 MB (AdamW exp_avg + exp_avg_sq). ≈355 MB per run, plus at most 177 MB of `.tmp` while saving. Kaggle's limit is 20 GB. |
| 5 | Two GPUs (T4 x2) | PASS | `torch.device("cuda")` = cuda:0 only; the second T4 sits idle. Harmless. |
| 6 | Data discovery | PASS | `rglob("fraudlm_sft_train.json")` under `/kaggle/input`, so any dataset slug or mount layout works. With more than one copy it prints all of them and uses the first. |
| 6 | Data sha check | PARTIAL → R2 | A sha is recorded and locked across resumes (`train_sft.py:345`), but it is never compared with the repo's current dataset. Today: train `4d7b29eef2740cd1`, val `87526bcd9d1c07df` (sha256[:16], files of 05:32). |
| 6 | Runbook | PASS with gaps → R4 | Upload, Add Input, commit, download and resume steps are right. Missing: a first-10-minutes check, a resume check, and the run-id change if G2 picks Large. One claim is unverified (output of a failed version). |
| 7 | Headers / comments / secrets / CLI | PASS | All 7 files have the PLAN §6 header; Raschka credit in `sft_data.py`, `train_sft.py` and the bundle. Grep for `hf_…`, `KAGGLE_KEY/USERNAME`, `api_key`, `token=`, `kaggle kernels/datasets` in the 7 files: no hits. |
| 7 | Bundle current | PASS | `test_committed_train_bundle_is_current` passes; header says generated 2026-11-06 06:01. |
| — | Deliverable names | ACCEPT | trainer.md asks for `kaggle/train_kernel.py` + `kernel-metadata.json` + `RUNBOOK.md`. The paste-in bundle + `RUNBOOK_G4.md` replaces them, and no metadata is needed without a CLI push. |

### fp16 simulation (exact GPU recipe replayed on CPU, gpt2-small)
Script: fp16 base via `load_pretrained_gpt2(dtype=float16)` → `apply_lora(16, 32)` → ckpt on → `batch_loss(..., torch.float16)`
under `torch.autocast("cpu", float16)` → `GradScaler("cpu", enabled=True)` → `scale(loss/len(group)).backward()` →
`unscale_` → `clip_grad_norm_` → `step` → `update`. Four short train rows, micro-batch 1 × accum 2, 2 steps.
- step 0: loss 5.2870, grad-norm 27.26, scale 65536→65536 (step applied)
- step 1: loss 4.1134, grad-norm nan, scale 65536→32768 (overflow: GradScaler skipped the step and halved the
  scale, as designed)
- LoRA params stay fp32 and change; frozen base stays fp16. The recipe raises no dtype errors.
- On CPU the residual stays **fp16** (block 0–2 in/out fp16, GELU fp16): CPU autocast does not upcast `pow`. The
  CUDA behaviour could not be measured here; see the analysis.
- The disabled-scaler path (`enabled=False`, CPU runs) is a no-op wrapper; checked separately (step applied,
  `state_dict()=={}`).

### fp16 analysis (item 2)
- **The main fp16 failure mode is silent, not NaN.** Our `LayerNorm` (`gpt_model.py:84-89`) is hand-written, so
  autocast does not upcast it (only `F.layer_norm` is on the fp32 list). With an fp16 residual, `x.var()` returns
  fp16. On CUDA it accumulates in fp32, but the result overflows to `inf` once the variance passes 65,504. Then
  `(x-mean)/sqrt(inf)` = 0, so that token's normalised row silently becomes `shift`: the loss is finite, just worse.
  The NaN guard would not catch it. gpt2-small already reaches a variance of 12,480 (5× headroom, from massive
  activations of ~3,000). XL is unmeasured. Every layer's LN sees the fp16 residual.
- Possible mitigating effect on CUDA (unverified here): `GELU` uses `x.pow(3)` (`gpt_model.py:96`), and the CUDA
  autocast op reference lists `pow` as fp32. If so, the GELU output is fp32, FF-out `LinearWithLoRA` returns fp32
  (`lora.py:43, 56`), and `x + shortcut` promotes the residual to fp32 from block 0 onward, which would remove
  the LN risk in training. On CPU this does **not** happen (measured: fp16 throughout). Do not rely on it. G2 can
  print `model.trf_blocks[1]` output dtype under autocast to settle it.
- Attention: SDPA kernels accumulate in fp32 and scale inside the kernel; T4 (sm75) uses mem-efficient/math. OK.
- Logits: fp16 out of the tied head (max abs ≈342 → fp16 step 0.25), then `.float()`. G1 measured a loss shift of
  0.0013 on gpt2-small with an fp16 base. Acceptable.
- GELU `x.pow(3)` without autocast (chat demo with an fp16 base) overflows for |x|>40, but the result stays
  finite and correct (tanh(±inf)=±1). No NaN.
- Residual precision: fp16 near 3,000 has a step of 2. This is the same as any `model.half()` GPT-2 and
  acceptable.
- **The real test is G2 on XL**: fp16-base vs fp32-base `sanity_loss` on the same text (R8). R3 catches a broken
  base in the first minutes of the run.

## Fixes (ordered by risk to the Kaggle run)

**R1 (required, crash on every GPU resume).** `training_model/train_sft.py:401`: load the resume state on the CPU:
```python
state = torch.load(run_dir / "train_state.pt", map_location="cpu", weights_only=False)
```
`opt.load_state_dict` moves the AdamW moments to each parameter's device by itself, and the scaler state is plain
floats. Optionally also `.cpu()` the two RNG tensors on lines 406/409. Add a unit test that loads a saved
`train_state.pt` through the same code path and asserts `state["rng_cpu"].device.type == "cpu"`. The kill/resume
test cannot see this on a CPU-only machine.

**R2 (required, silent training on stale data).** The dataset is still in flux: F10 (age-proxy) may trigger a
rebuild after Thông has uploaded v1.1. Bake the expected hash into the bundle and check it:
`make_bundle.py` writes `EXPECTED_DATA_SHA256 = "<train16>-<val16>"` (computed from `dataset/` at build time) into
Part C. In `train_sft.py`, after line 345: if it is set and differs from `cfg["data_sha256"]`, raise
`SystemExit("[data] Kaggle dataset differs from the repo dataset: upload a New Version or rebuild the bundle")`
unless `--allow_data_mismatch` is given. Also print the sha on the `[data]` line. Add a bundle test that checks the
baked value equals the current files.

**R3 (strongly advised, cheap guard against a broken fp16 base wasting hours).** `train_sft.py:458-462`: after
the step-0 eval, if `not math.isfinite(v0) or v0 > 5.0`, write samples and raise
`SystemExit("[eval] step-0 val loss {v0} is not sane for pretrained GPT-2: try base_dtype fp32")`. That costs
about 5 min instead of a full session. Reference: gpt2-small step-0 val-subset loss in the smoke test is 2.60, and
XL should be lower. Do the same on the `[nan] STOPPED` path. Keep `eval_at_start` on for the Kaggle run.

**R4 (runbook, `kaggle/RUNBOOK_G4.md`).**
- Add a "first 10 minutes" check after step 3.3. Open the log and confirm: `[gpu] Tesla T4 …`;
  `[data] train: 12,000 examples … cap 512` with no `WARNING`; the data sha (R2); `[plan] … 3,000 steps total` (or
  G2's value); `[eval] step 0: val subset loss` finite and about 2–3; s/step within ~20% of G2. If any is wrong,
  **Cancel** the version.
- Resume row (line 59): after relaunching, confirm `[resume] from step N`. If the log shows `[eval] step 0`
  instead, the input was not found: cancel it, or the run restarts from scratch and the 11 h are spent again.
- Line 60: "Version failed … Output tab" is unverified. Do not promise that a failed or timed-out commit keeps
  `/kaggle/working`. The `--max_hours` guard (a clean, "complete" version) is the supported path.
- Lines 46-49, 59: the paths hard-code `xl_r16_v1`. If G2 picks Large, the lead must change `run_id` in DEFAULTS
  and these paths together.
- Lead's pre-launch step: put the G2 values (micro_batch, grad_accum, grad_ckpt, base_dtype, size) in DEFAULTS,
  re-run `make_bundle.py`, and run `pytest tests/test_train_smoke.py` (the bundle-current test).

**R5 (low, resume consistency).** `train_sft.py:434-446` writes `lora_adapter.pt` and then `train_state.pt`. A
kill between the two leaves the adapter at step N and the optimiser/progress at N−200. The resume then silently
replays 200 steps on top of the step-N adapter. Fix: put the adapter tensors inside `train_state.pt` as well and
restore from there (keep `lora_adapter.pt` as the deliverable), or store `step` in both files and refuse on a
mismatch.

**R6 (low, document it).** Accumulation averages the micro-batch means (`train_sft.py:496`). With length bucketing,
tokens in short micro-batches weigh more than tokens in long ones. This is acceptable (Raschka does the same per
batch). Report it as "token-mean per micro-batch, averaged over 2 micro-batches", or switch to `reduction="sum"`
divided by the learned-token count of the whole group, which is known before the forward pass.

**R7 (low, observability).** GradScaler skips are invisible. Add `scaler.get_scale()` to the step line and count
skipped steps (the scale went down after `update()`) into `progress` and `run_summary.json`. If more than ~5% of
steps after warm-up are skipped, or the scale falls below 1, the run is not learning, and the log should say so.

**R8 (advised before launch: G2 acceptance + model-builder insurance).** (a) Lead: on XL, compare G2's fp16-base
`sanity_loss` with its fp32-base `sanity_loss` on the same text; accept fp16 only if they differ by less than
0.05. Otherwise use `base_dtype fp32` (6.2 GB of weights; autocast still uses fp16 matmuls). (b) Model-builder,
5 lines, removes the silent LN-overflow mode in training **and** in the chat demo: compute
`LayerNorm.forward` (`gpt_model.py:84-89`) in fp32:
```python
xf = x.float()
mean = xf.mean(dim=-1, keepdim=True)
var = xf.var(dim=-1, keepdim=True, unbiased=False)
norm_x = (xf - mean) / torch.sqrt(var + self.eps)
return (self.scale.float() * norm_x + self.shift.float()).to(x.dtype)
```
This is bit-identical for fp32 inputs, so all G1 tests still hold. If it lands, rebuild both bundles (G2 and G4)
before the G2 run, so that G2 measures this code.

## Notes for other agents
- Chat-builder/evaluator: on the CPU, use an fp32 base (the fp16 path is very slow there, and R8(b) does not apply
  until it lands). On a GPU, keep the adapter unmerged or merge it in fp32 (G1 note).
- Explainability: the code is plain and commented. Thông should be able to explain why the loss is shifted only
  once (inputs/targets are built in `encode_example`), and what GradScaler's skipped steps mean.
- Simulation scripts (reviewer scratch, not in the repo): masked-loss re-derivation, padding, magnitudes, ckpt
  equality, RNG device check, kernel argv, fp16 recipe replay.

---

## Addendum: re-check of the fixes (reviewer, 2026-11-06 06:49 UTC+7)

Scope: only the changed lines in `training_model/train_sft.py`, `training_model/gpt_model.py`,
`kaggle/make_bundle.py`, `kaggle/feasibility.py`, both runbooks, the new tests, and the two rebuilt bundles.
Reviewer run: `.venv/Scripts/python -m pytest tests -q` → **106 passed in 76.15s**.

### Verdict for launch: **PASS**. G4 can be launched once the G2 values are in `DEFAULTS` and the bundle is rebuilt

| R | Result | Evidence |
|---|---|---|
| R1 RNG on CPU | Resolved | `train_sft.py:232-242` `torch.load(map_location="cpu")`, RNG keys forced to `.cpu().to(uint8)`; used at line 458. Optimiser `load_state_dict` moves the moments to each param's device; `load_lora` uses `copy_` (`lora.py:118`), so CPU tensors are fine. Tests `test_load_train_state_moves_rng_to_cpu`, `test_train_state_loads_on_cpu_with_uint8_rng_and_holds_the_adapter` pass. |
| R2 data sha baked + abort | Resolved | `kaggle/train_bundle.py:1152` `EXPECTED_DATA_SHA256 = "4d7b29eef2740cd1-87526bcd9d1c07df"`; the reviewer recomputed the repo `dataset/` hash and got the same value. Mismatch → `SystemExit` before the model loads (`train_sft.py:388-394`) unless `--allow_data_mismatch`. The repo copy stays empty, so the check is off locally (line 128). `make_bundle.py:58,73-74` replaces the marker. Tests `test_data_hash_mismatch_stops_the_run`, `test_committed_train_bundle_has_current_data_hash` pass. |
| R3 step-0 guard | Resolved, design accepted | `check_start_val_loss` (`train_sft.py:245-250`) raises at step 0 (non-finite or > 5.0), after the step-0 log row is written and before any update. Raising here costs nothing, because nothing is trained yet. Returning cleanly from the NaN stop (`train_sft.py:556-566`, status "nan", exit code 2 outside a notebook) is the right choice: the version ends "complete", `/kaggle/working` keeps the last good `train_state.pt`/`lora_adapter.pt`, no checkpoint is written after the NaN, and the runbook table covers the banner. |
| R4 runbook + run_id | Resolved | `train_sft.py:191-192` gives `"<size minus gpt2->_r<rank>_v1"` (test `test_run_id_follows_size`). RUNBOOK_G4 §3.3 first-10-min table (GPU, `= bundle (OK)`, examples, steps, step-0 loss 2–3, s/step); the §5 resume row checks `[resume] from step N` (if the log shows `[eval] step 0`, cancel); the failed-version claim is now qualified; `<run_id>` is used in the paths. |
| R5 adapter + state atomic | Resolved | `save_checkpoint` (`train_sft.py:489-506`) writes `lora` inside `train_state.pt` with one `os.replace`; resume reads `state["lora"]` (line 456). `progress.json` is written last; `find_resume_dir` only uses it to rank copies, and the step comes from `train_state.pt`, so a kill between the writes stays consistent. |
| R6 comment | Resolved | `train_sft.py:541-545`. |
| R7 scaler columns | Resolved | `LOG_COLUMNS` = the 7 PLAN §3 contract columns, then `grad_scale, amp_skipped` (`train_sft.py:122-123`). Rows at lines 520 and 606-608 match the order. A skip is counted from the scale drop (`train_sft.py:583-588`), which is correct for GradScaler's backoff (growth only raises it). `[amp] WARNING` after warmup+50 steps if >5% are skipped or scale < 1 (`train_sft.py:639-647`). |
| R8 LayerNorm | Resolved | `gpt_model.py:84-96`: computes in `promote_types(x.dtype, fp32)`, then casts back. Reviewer: rows ×3000 in fp16 come out finite, max diff 5.2e-4 vs fp32; fp32 in → fp32 out, bf16 in → bf16 out. Present in both bundles (`train_bundle.py:116-137`, `feasibility_bundle.py:116-137`). Both bundles are stamped 06:44, after the model-builder change at 06:41, and the bundle-is-current tests pass. |
| R8 G2 fp16-vs-fp32 check | Resolved | `feasibility.py:350-371`: fp32 base is loaded first as the reference; fp16 is accepted only if \|diff\| < 0.05 (non-finite → inf → rejected); if fp16 is rejected and fp32 was not timed, fp32 gets timed. `fp16_checks` goes into the JSON (line 515). |
| Hook does not alter outputs | Confirmed | `feasibility.py:171-180`: the hook returns `None` and is removed before the loss. Reviewer: a 3-layer GPTModel gives logits with and without the hook that differ by **0.0**, and the dtype is recorded. |

Non-blocking notes (no change needed for launch):
1. The fp32-base "reference" in `feasibility.py` also runs under fp16 autocast, so it isolates the fp16 *base
   storage / fp16 residual* question, which is the one R8 asked about. An autocast-wide breakage would hit both
   sides; the step-0 val-loss guard (< 5) in training catches that case.
2. Launch preconditions, unchanged: G2 values (size, base_dtype from the `[fp16 check]` verdict, grad_ckpt,
   micro_batch, grad_accum) go into `DEFAULTS`, then `python kaggle/make_bundle.py`, then `pytest tests -q`
   all green, before Thông pastes the bundle. If `dataset/` changes again, the bundle must be rebuilt and the
   Kaggle dataset re-versioned (R2 enforces this).
