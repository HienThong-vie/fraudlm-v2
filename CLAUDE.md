# llm_ver2: FraudLM v2 (GPT-2 + LoRA route)

This folder replaces `../llm/` for all LLM work. **`PLAN.md` here overrides `../llm/BUILD_PLAN.md` and
`../llm/AGENTS.md`**: the old from-scratch 300M pretraining spec, its 06/11 12:00 deadline and its "no
LLM-generated data" rule no longer apply. The parent `../CLAUDE.md` still applies for everything else
(A2 protocol, protected attributes, test months 6–7, no generative-AI prose in submitted documents).

- Read `PLAN.md` (decisions, folder ownership, interface contracts, gates) before any work.
- Team of agents: `.claude/agents/` (model-builder, data-builder, trainer, evaluator, chat-builder, reviewer).
  The main session is the lead: it assigns jobs, keeps `PLAN.md`, and merges results.
- Progress log: `TEAM_LOG.md` (append only).
- Reusable pieces from the old route (read, adapt, never import from `../llm/`): SFT generators
  `../llm/src/fraudlm/sft/`, chat guard `../llm/src/fraudlm/infer/guard.py`, chatbot metrics
  `../llm/BUILD_PLAN.md` Section 9.2, dry-run SFT data `../llm/data/sft_dryrun/`.
- Local PC has no CUDA: develop and test on CPU with `gpt2-small`; GPT-2 XL runs only on Kaggle.
