# P-EAGLE-Q: A Characterization Study of Quantization Mismatch in EAGLE-Style Speculative Decoding

_Last updated: April 16, 2026_

This revision replaces the earlier "recovery-only" framing. The pilot data at 1k training examples falsified the original H1 for W4A16 weight-only quantization, so the paper now asks a sharper and more general question: **under which quantization regimes does train-vs-inference mismatch actually hurt EAGLE-style speculative decoding, and when quantization-aware draft training helps, how much of the gap does it close?**

## Evidence Tags

Use the following tags consistently throughout the paper draft:

- **Observed**: directly stated in a cited paper, repository, release note, pull request, or a reproducible experiment from this project.
- **Inferred**: a conclusion drawn from observed facts, but not directly stated by a source.
- **Hypothesized**: a claim that still requires experiments.

## 0. Current Status (April 16, 2026)

### 0.1 Pilot Results at 1k Training Examples

The three Qwen3-8B rows from the 1k pilot:

| Row | Drafter trained on | Verifier | τ | α[0] | α[1] | α[4] |
| --- | --- | --- | --- | --- | --- | --- |
| (a) BF16 → BF16 | BF16 teacher | BF16 | 1.648 | 0.663 | 0.395 | 0.113 |
| (b) BF16 → AWQ W4A16 | BF16 teacher | AWQ W4A16 | 1.645 | 0.667 | 0.400 | 0.116 |
| (c) AWQ → AWQ W4A16 | AWQ W4A16 teacher | AWQ W4A16 | 1.627 | 0.666 | 0.392 | 0.114 |

- **Observed**: Row (a) and row (b) differ by less than 1.3% on τ and less than 0.5 percentage points on α[0]. The BF16-trained drafter does **not** degrade on the W4A16 verifier.
- **Observed**: Row (c) performs within noise of row (b). The quantization-aware drafter does not improve on the BF16-trained control for this regime.
- **Inferred**: Weight-only quantization (W4A16) does not introduce measurable drafter-facing distribution shift, because drafter tap-layer outputs remain in FP16/BF16 on both sides of the pipeline.
- **Inferred**: The original "recovery" framing cannot be supported by these results. The paper needs a different thesis.

### 0.2 Known Implementation Issues

- **Observed (April 16, 2026)**: Several correctness issues were fixed before the 8k chain: unresolved `$VAR` config expansion now raises, cosine decay replaces a flat-after-warmup LR, teacher hidden-state dtype is no longer silently cast to FP16, `strict=True` and `weights_only=True` are enforced on checkpoint load, and a `seed` field is plumbed through `TrainRunConfig`.
- **Resolved (April 17, 2026)**: The speculative evaluator in `peagle_q/eval.py` has been restructured into two paths controlled by `speculation_mode` on `EvaluationConfig`:
  - `sequential` — KV-cached, 2 verifier forwards per speculative step. This is the matched-control path for τ/α and is the default for the 1 k pilot configs.
  - `parallel` — real parallel-verification speculative decoding. The draft head produces K tokens autoregressively via `Eagle3StyleDraftHead.propose_sequence` (self-feeding the draft's own output hidden across the tap axis in lieu of per-position verifier hiddens), then the verifier verifies all K in a single forward plus one tip-extension forward. This path produces meaningful `tokens_per_second` and `speedup_vs_ar`.
  - `both` — run each prompt through both paths in one evaluator pass and write a `by_mode` section in the summary; recommended for the 8 k matrix. Cross-path `τ` comparison also quantifies the acceptance cost of the self-feeding approximation.
- **Inferred**: The 1 k pilot τ/α numbers stand unchanged (accept/reject counts are not affected by the timing fix). `speedup_vs_ar` values in the 1 k JSONs remain uninterpretable, but new 8 k eval configs default to `speculation_mode: "both"`, which writes real speedup figures.

### 0.3 What This Status Tells Us

- **Inferred**: The pilot has produced a genuine scientific finding (the "W4A16 null cell"). It is just not the finding the original plan predicted.
- **Inferred**: The correct move is to widen the study to other quantization regimes where activations or the KV cache are quantized, which is where drafter-facing drift should theoretically appear.

## 1. Scoped Position

- **Observed**: The existing literature documents EAGLE-style draft training and quantized speculative inference as separate topics, and documents specific failure modes in mixed precision deployments (SpecMQuant, vLLM PR #25883).
- **Observed**: The 1k Qwen3-8B pilot in this project shows that W4A16 (AWQ) is not a regime in which EAGLE-style acceptance degrades.
- **Inferred**: The open question is not "does quantization hurt speculative decoding," which is too broad, nor "does QAT draft training recover W4A16 loss," which our own data rules out. The open question is **where the mismatch actually lives and what, if anything, recovers it.**

This is the claim the paper should make in v2:

> Quantization-induced acceptance loss in EAGLE-style speculative decoding is not uniform across quantization regimes. Weight-only regimes (W4A16) do not measurably hurt acceptance on a matched benchmark. Activation-quantized regimes (W8A8, FP8 all-quantized, KV4 cache) are the cells where mismatch should appear, and are therefore the cells where quantization-aware draft training can be tested as a recovery method.

This is the claim the paper should **not** make:

> "Quantization-aware draft training universally improves speculative decoding on quantized verifiers."

That form of the claim is already falsified by the 1k pilot for the W4A16 cell.

## 2. Anchor Evidence Snapshot

The table below is a dated snapshot, not a final literature appendix.

| Source | Dated fact | What it directly supports | Relevant cell |
| --- | --- | --- | --- |
| [P-EAGLE paper page](https://papers.cool/arxiv/2602.01469) | Published February 1, 2026 UTC | Parallel-drafting EAGLE training exists at scale | Baseline drafter architecture |
| [SpecMQuant / CoRR abs/2505.22179](https://dblp.org/rec/journals/corr/abs-2505-22179.html) | 2025 CoRR record | Quantized speculative decoding compatibility is a real problem; hierarchical inference proposed | Activation-quantized cells |
| [vLLM Speculators repository](https://github.com/vllm-project/speculators) | Accessed April 9, 2026 | Offline hidden-state generation, draft training, deployment tooling in one stack | Infrastructure |
| [vLLM PR #25883](https://github.com/vllm-project/vllm/pull/25883) | Merged September 29, 2025 | Quantized-verifier + unquantized-drafter inference is a supported scenario | W4A16 cell |
| [llm-compressor](https://github.com/vllm-project/llm-compressor) | Accessed April 16, 2026 | Public implementation of W8A8, W4A8, FP8 quantization recipes | Activation-quantized cells |
| [KIVI / KV-quant literature](https://arxiv.org/abs/2402.02750) | 2024 | KV-cache quantization introduces activation-path drift | KV-quant cells |

- **Observed**: Anchor sources support the infrastructure and the W4A16 inference scenario.
- **Inferred**: No anchor source has published a dated end-to-end table of drafter acceptance across (W4A16, W8A8, KV4, FP8) with a matched-control protocol on a single drafter family.
- **Inferred**: This is where the paper's contribution can sit honestly, without claiming the literature is "exhaustively closed."

## 3. Claims We Can Defend Now

- **Observed**: EAGLE-3-style drafters can be trained offline from hidden states extracted by either a BF16 or an AWQ W4A16 teacher on Qwen3-8B.
- **Observed**: At 1k training examples and 20-question MT-Bench subset, a BF16-trained drafter evaluated on an AWQ W4A16 verifier matches its BF16-on-BF16 row within noise.
- **Observed**: At 1k training examples, a W4A16-teacher-trained drafter does not beat the BF16 control on a W4A16 verifier.
- **Inferred**: W4A16 is the correct "null cell" of the study; the paper's contribution cannot rely on this cell.
- **Inferred**: The existence of the null cell is informative: it tells practitioners they do not need QAT draft training for AWQ-style weight-only deployments.

## 4. Claims That Still Require Experiments

The hypotheses are restructured around the new thesis. All hypotheses assume:

- Qwen3-8B as the primary model family.
- Llama-3.1-8B as a secondary transfer target for at least H1 and H2.
- Matched data, tokenizer, seed, optimizer, sequence length, epoch budget, drafter parameter count, and speculative budget across all training runs within a given cell comparison.

### H1 (confirmatory, mostly already run). Weight-only quantization does not introduce measurable drafter-facing drift.

- **Hypothesized** (already observationally supported by the 1k pilot): A BF16-trained drafter evaluated on a W4A16 (AWQ) verifier does not lose acceptance relative to the BF16→BF16 row, and the layer-wise hidden-state drift between BF16 and AWQ teachers is small.
- **Working success threshold**: At 8k training examples, |α[0] delta| < 2 percentage points on the primary benchmark, and mean per-layer CKA between BF16 and AWQ hidden states at drafter tap points is ≥ 0.98. Repeat on Llama-3.1-8B as a transfer check.

### H2 (core new hypothesis). Activation-quantizing regimes introduce measurable drift and acceptance loss.

- **Hypothesized**: A BF16-trained drafter evaluated on a W8A8 SmoothQuant verifier (and separately, an FP8-all-quantized verifier, and a BF16-weight + KV4 verifier) loses acceptance relative to the BF16→BF16 row.
- **Working success threshold**: At least one of the three activation-quantized cells shows α[0] drop ≥ 5 percentage points, with paired 95% CI excluding zero on per-prompt deltas.
- **Kill rule**: If no activation-quantized cell shows a measurable drop, revert to the null-result framing (see §11).

### H3 (recovery, conditional on H2). Quantization-aware draft training recovers some of the H2 loss.

- **Hypothesized**: For any activation-quantized cell where H2 is confirmed, a drafter trained from that cell's hidden states beats the matched BF16-trained control on the same verifier.
- **Working success threshold**: Recovers ≥ 25% of the α[0] loss or improves end-to-end ITL by ≥ 5%, while preserving exact-output equivalence with autoregressive decoding on the same verifier.

### H4 (mechanism). Layer-wise drift predicts acceptance loss.

- **Hypothesized**: Across cells, mean CKA (or mutual-information proxy) between BF16 and quantized teacher hidden states at drafter tap layers correlates with |α delta| at position 0.
- **Working success threshold**: Spearman ρ ≥ 0.6 across ≥ 6 cells (spanning W4A16, W8A8, FP8, KV4, and any negative controls), with the correlation figure carrying the paper's mechanism story.

These thresholds are proposal-stage. They should be revisited after the first activation-quantized run.

## 5. Canonical Implementation Stack

Use one canonical stack for training and evaluation. Add a quantization toolchain for each new cell, but keep the drafter and evaluation code fixed.

- Drafter training and evaluation: the current `peagle_q` repo (transformers-based teacher runner).
- Model family: Qwen3-8B primary. Llama-3.1-8B-Instruct as a secondary transfer family.
- Drafter architecture: EAGLE-3-style multi-tap fused head, held constant across cells.
- Quantization toolchain per cell:
  - W4A16 → `autoawq` (done).
  - W8A8 → `llm-compressor` (SmoothQuant + GPTQ-INT8 activations).
  - FP8 all-quantized → `torchao` FP8 recipe if H100 access is available; otherwise mark as deferred.
  - KV4 cache → either the verifier-side KV-quant path in vLLM, or a custom KV-cache quantizer in the transformers runner. Pick one and document.

Contingency: if any quantization backend is not installable on Sol, the cell is marked deferred and does not enter the main comparison table.

### Compute path

- Use Sol `4x A100 80 GiB` nodes for the main matched-control training runs. A single A100 80 GiB is enough for 8B-class verifier inference and 8k-example training.
- Use Sol scratch (`$SCRATCH`) for all large intermediate artifacts.
- Keep Gemma 1B only for local smoke tests; do not enter the main comparison.
- Treat Llama-3.1-8B-Instruct as a transfer-family check, not a replacement for Qwen3.

## 6. Fairness-Tight Experimental Contract

The core matrix has two axes: **what gets quantized** (W4A16, W8A8, FP8, KV4) and **what the drafter trained on** (BF16 teacher, matched-quantized teacher). Every cell in the main table uses freshly trained matched controls.

### Required training runs per model family

- BF16-control drafter (already exists for Qwen3-8B at 1k; retrain at 8k, multi-seed).
- Per activation-quantized cell: one drafter trained from that cell's teacher hidden states, under the same hyperparameters as the BF16 control.

### Required evaluation configs

| Config | Drafter trained on | Verifier weights | Verifier activations | KV cache | Role |
| --- | --- | --- | --- | --- | --- |
| AR BF16 baseline | None | BF16 | BF16 | FP16 | Upper-bound verifier reference |
| BF16 drafter → BF16 | BF16 | BF16 | BF16 | FP16 | Row (a) — matched control |
| BF16 drafter → W4A16 | BF16 | W4 | FP16 | FP16 | Row (b) — null-cell confirmation |
| W4A16 drafter → W4A16 | W4A16 | W4 | FP16 | FP16 | Row (c) — null-cell ceiling |
| BF16 drafter → W8A8 | BF16 | W8 | INT8/FP8 | FP16 | H2 test cell 1 |
| W8A8 drafter → W8A8 | W8A8 | W8 | INT8/FP8 | FP16 | H3 recovery cell 1 |
| BF16 drafter → KV4 | BF16 | BF16 | BF16 | INT4 | H2 test cell 2 |
| KV4 drafter → KV4 | KV4 | BF16 | BF16 | INT4 | H3 recovery cell 2 |
| BF16 drafter → FP8 all | BF16 | FP8 | FP8 | FP8 | H2 test cell 3 (deferred if no H100) |
| FP8 drafter → FP8 all | FP8 | FP8 | FP8 | FP8 | H3 recovery cell 3 (deferred if no H100) |

### Rules for keeping the comparison clean

- Prompt formatting, tokenizer, decoding settings, speculative token budget, and benchmark prompts stay fixed across all rows.
- Drafter parameter count stays constant across all rows in a given model family.
- Each drafter training run uses three seeds; report mean and seed variance.
- Speculative outputs must match autoregressive outputs from the same verifier on the same prompts at temperature 0.
- The same extracted dataset (same prompts, same sequence length cap, same chat template) is used for every drafter training run within a family.

## 7. Mechanism Diagnostics

These diagnostics are required for all cells in the main table. They carry the paper regardless of which direction H2 resolves in.

- Per-layer hidden-state drift at drafter tap points: CKA(BF16 teacher, quantized teacher) on matched prompt tokens.
- Per-layer logit agreement between BF16 and quantized verifiers: top-1 agreement and KL divergence.
- Acceptance by speculative depth (α per position), not just aggregate τ.
- Exact-output equivalence checks between speculative and autoregressive decoding on the same verifier.
- Latency decomposition into drafting time, verifier time (parallel-verification path), and end-to-end latency.
- Self-feeding approximation cost: τ(sequential) − τ(parallel) on the same drafter, reported per cell.

### Interpretation rules

- **Observed**: A cell where the quantized verifier is merely slower, without any drop in acceptance or measurable drift, is a systems cell, not a teacher-mismatch cell.
- **Inferred**: A causal training-mismatch story is supported only if (acceptance drops) ∧ (drift rises) ∧ (QAT draft training recovers part of the loss).

## 8. Evaluation and Reporting

### Primary metrics

- Mean accepted length τ (reported from the sequential path as the matched-control metric).
- Per-position acceptance α[k] (same path).
- Inter-token latency (ITL) and time to first token (TTFT) (reported from the parallel-verification path).
- End-to-end latency relative to the matched autoregressive baseline (parallel path).
- Self-feeding approximation cost: τ(sequential) − τ(parallel) reported alongside the speedup row so a reader can price the quality/throughput trade.

### Secondary metrics

- Theoretical-ideal speedup (τ + 1) as a drafter-quality upper bound, independent of verifier speed. Already added to the evaluator output.
- Verifier-side CKA and logit KL, reported in the mechanism figure.

### Benchmarks

- Primary: MT-Bench, full 80 questions (not the 20-question mini benchmark — that was only for pipeline validation).
- Secondary: at least one additional benchmark with enough prompts for paired confidence intervals. Candidates: GSM8K (reasoning), HumanEval (code), SlimPajama-Chat or similar (dialog). Pick one and justify.

### Statistical reporting

- Paired prompt-level deltas with 95% CIs for every row comparison.
- Seed variance across three training seeds per drafter.
- The core results table reports mean ± 95% CI, not point estimates.

The core result should be framed as:

> We show that mismatch between the drafter's training teacher and the deployed quantized verifier is regime-dependent: weight-only quantization (W4A16) is a null cell; activation-quantizing regimes show a measurable acceptance gap; quantization-aware draft training closes part of that gap in the affected cells, with mechanism diagnostics supporting a drift-based explanation.

Not as:

> Quantization-aware training universally solves speculative decoding under quantization.

## 9. Kill Criteria and Pivot Paths

- If H1 fails at 8k (i.e., the W4A16 null result does not hold at larger scale), revisit extraction code, seed variance, and benchmark size before concluding the pilot was a fluke.
- If H2 fails across all activation-quantized cells, the paper becomes a null-result paper: **"EAGLE-style drafters are robust across common quantization regimes"**, anchored by the mechanism figure. This is still publishable, at workshop scale.
- If H2 holds but H3 fails, the paper becomes **"Where speculative decoding breaks under quantization, and why training the drafter on quantized teachers is not enough to fix it."** This is a stronger negative result and still publishable.
- If H2 and H3 both hold, the paper is a full characterization study. Target a main conference (ACL, EMNLP, NeurIPS datasets/benchmarks track).
- If parallel-path latency numbers look interesting, the paper adds a systems section, but latency does not carry the paper on its own.

## 10. What This Draft Intentionally Removes from the Core Story

- Any claim framed as "quantization-aware training recovers lost performance" without naming the regime.
- Any latency claim not produced by the parallel-verification path (`speculation_mode="parallel"` or `"both"`). The sequential-path `speedup_vs_ar` remains defined only for matched-control comparison and is not a systems claim.
- Predicted acceptance values, projected speedups, training-hour estimates, and venue-specific deadlines. Those belong in project notes, not the paper draft.
- The mini 20-question benchmark as a headline result. It remains only as a pipeline sanity check.

## 11. Deliverables for the Next Revision

Concrete artifacts that should exist before the next paper revision:

- The `eval.py` 6-verifier-pass-per-step bug is fixed and the pre-fix vs post-fix latency delta is documented.
- 8k training runs finished for the BF16 control and the AWQ W4A16 drafter on Qwen3-8B, with three seeds each. This confirms or overturns H1 at scale.
- At least one activation-quantized cell (preferred order: W8A8 SmoothQuant, then KV4, then FP8) has both its BF16-control row and its matched-QAT row run.
- A mechanism figure with per-layer CKA and logit KL between BF16 and each quantized teacher, aligned with per-cell α deltas.
- A Llama-3.1-8B transfer run for H1 at minimum (ideally also one H2 cell).
- A novelty-audit appendix with a dated search protocol, inclusion and exclusion criteria, and a source table reproducible by another reader.

## 12. Sol Execution Plan (Revised)

### Phase A: confirm the null cell at 8k (currently staged)

- Run the 8k chain for Qwen3-8B: extract BF16, extract AWQ, train BF16 drafter (5 epochs, seed 42), train AWQ drafter (5 epochs, seed 42), evaluate rows (a), (b), (c) on the full MT-Bench (80 questions).
- Success: τ delta between rows (a) and (b) stays below 2 percentage points on full MT-Bench, with CIs excluding a > 5-point drop. This formally closes H1.
- Add two more seeds (1, 7) for each drafter. This gives the seed-variance number the paper needs.

### Phase B: fix the evaluator systems bug

- Refactor `speculative_generate_local` to issue a single verifier forward pass per speculative step that returns both tap-layer hidden states (for the next proposal) and verification logits.
- Re-run row (a) as a regression: τ and α must stay unchanged; ITL and tokens-per-second should increase materially.
- Only after this fix can any latency column enter the paper.

### Phase C: add W8A8 (SmoothQuant) cell on Qwen3-8B

- Install and validate `llm-compressor` on Sol. Produce a W8A8 Qwen3-8B checkpoint.
- Run a smoke test: autoregressive generation works, hidden-state extraction works.
- Extract W8A8 teacher hidden states at 1k, then 8k.
- Train a W8A8-aware drafter (3 seeds).
- Evaluate: BF16 drafter → W8A8 (H2 test), W8A8 drafter → W8A8 (H3 recovery).
- Decision point: does the BF16 drafter drop ≥ 5 points on α[0] against the W8A8 verifier?

### Phase D: add KV4 cache cell

- Implement or adopt a KV4 cache quantizer on the verifier side that does not change weight precision.
- Extract teacher hidden states with KV4 active so the drafter sees the true deployed distribution.
- Train a KV4-aware drafter (3 seeds).
- Evaluate H2 and H3 for this cell.

### Phase E (optional): add FP8 all-quantized cell

- Only if H100 access becomes available, or if FP8-all emulation on A100 is acceptable for the paper's purpose.
- Otherwise mark as deferred and state so in the paper.

### Phase F: transfer check on Llama-3.1-8B

- Repeat Phase A and at least one activation-quantized phase on Llama-3.1-8B-Instruct.
- If Llama access is blocked, document and continue with Qwen3 only, noting the single-family limitation.

### Phase G: mechanism figure

- For each teacher in (BF16 Qwen3, AWQ Qwen3, W8A8 Qwen3, KV4-active Qwen3, BF16 Llama, matched Llama cells), compute per-layer CKA and logit KL against the BF16 teacher on a fixed prompt set.
- Cross-plot against α[0] deltas for the corresponding cells.
- This is the paper's main mechanism figure.

## 13. Sol Pre-Flight Checklist

Run through this checklist on Sol **before** submitting `submit_8k_pilot.sh` (new phases below will need sibling scripts for W8A8 and KV4).

### Environment variables

```bash
export PEAGLE_ROOT=/path/to/this/repo
export PEAGLE_BF16_MODEL=Qwen/Qwen3-8B
export PEAGLE_AWQ_MODEL=Qwen/Qwen3-8B-AWQ
export PEAGLE_W8A8_MODEL=   # set once llm-compressor produces a checkpoint
export PEAGLE_DATASET=/path/to/sharegpt_or_other.jsonl
export PEAGLE_RUN_ROOT=$SCRATCH/peagle-q/$USER
export PEAGLE_BENCHMARK=$PEAGLE_ROOT/benchmarks/mt_bench_full.jsonl
export PEAGLE_CACHE_ROOT=$PEAGLE_RUN_ROOT/cache
```

### Step-by-step verification

| # | Check | Command | Pass criteria |
|---|-------|---------|---------------|
| 1 | Build venv | `bash scripts/sol/setup_env.sh` | Prints "Environment ready" with correct paths |
| 2 | Python version | `python3 --version` (inside venv) | >= 3.10 |
| 3 | Core imports | `python3 -c "import peagle_q; import torch; print(torch.cuda.is_available())"` | `True` |
| 4 | AWQ import | `python3 -c "from awq import AutoAWQForCausalLM; print('OK')"` | `OK` |
| 5 | llm-compressor import (Phase C) | `python3 -c "import llmcompressor; print('OK')"` | `OK` once Phase C starts |
| 6 | Dataset exists | `wc -l $PEAGLE_DATASET` | >= 8000 lines for the 8k chain |
| 7 | Full MT-Bench exists | `wc -l $PEAGLE_BENCHMARK` | == 80 questions |
| 8 | BF16 smoke | `bash scripts/sol/run_cli.sh smoke --config configs/sol/qwen3_8b_smoke.json` | JSON with `hidden_state_shape` and `autoregressive.text` |
| 9 | AWQ smoke | `bash scripts/sol/run_cli.sh smoke --config configs/sol/qwen3_8b_smoke_awq.json` | Same as above |
| 10 | Scratch space | `df -h $SCRATCH` | At least 200 GB free for the expanded matrix |
| 11 | Slurm access | `sinfo -p YOUR_PARTITION` | GPU partition visible |
| 12 | Evaluator bug status | `grep -n "forward_ids" peagle_q/eval.py` | One verifier pass per speculative step (Phase B done) before any latency claim |

### Submit the full 8k chain

```bash
bash scripts/sol/submit_8k_pilot.sh --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

### Expected outputs (per seed, per family)

```
$PEAGLE_RUN_ROOT/qwen3_8b/
├── extract_bf16_8k/
├── extract_awq_8k/
├── train_bf16_8k_seed42/
├── train_bf16_8k_seed1/
├── train_bf16_8k_seed7/
├── train_awq_8k_seed42/
├── train_awq_8k_seed1/
├── train_awq_8k_seed7/
├── eval_bf16_on_bf16_seed42.json
├── eval_bf16_on_awq_seed42.json
├── eval_awq_on_awq_seed42.json
└── ... (per seed)
```

### Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `config contains unresolved environment variable` | Env var not exported into the batch env | Export + `echo >> ~/.bashrc` + resubmit |
| `ModuleNotFoundError: awq` | `autoawq` not installed in venv | `pip install autoawq` inside `.venv-sol` |
| `ModuleNotFoundError: llmcompressor` | Phase C not installed yet | Install only when entering Phase C |
| AWQ smoke hangs or OOMs | Node too small | Confirm A100 80 GB, not A30 |
| Extraction produces 0 tensors | `$PEAGLE_DATASET` is a directory, not a file | Point to the `.jsonl` path |
| Training crashes on projection load | `lm_head` not on quantized checkpoint | Set `projection_model_path` to the matched BF16 model |
| Eval shows `tau: 0.0` | Draft checkpoint missing or empty | Verify `train_*/checkpoints/best.pt` exists |
| Speedup ≪ 1 | 6-verifier-pass bug still present | Apply Phase B fix before trusting latency |

## 14. Anchor Sources

- [P-EAGLE paper page](https://papers.cool/arxiv/2602.01469)
- [SpecMQuant / CoRR abs/2505.22179](https://dblp.org/rec/journals/corr/abs-2505-22179.html)
- [vLLM Speculators repository](https://github.com/vllm-project/speculators)
- [vLLM PR #25883](https://github.com/vllm-project/vllm/pull/25883)
- [llm-compressor](https://github.com/vllm-project/llm-compressor)
- [KIVI (KV-quant literature)](https://arxiv.org/abs/2402.02750)
