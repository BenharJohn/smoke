# P-EAGLE-Q: Quantization-Aware Draft Training as a Defensible Research Proposal

_Last updated: April 9, 2026_

This draft is intentionally narrower than the original concept note. It is written to survive reviewer scrutiny before any strong novelty or mechanism claim is made.

## Evidence Tags

Use the following tags consistently throughout the paper draft:

- **Observed**: directly stated in a cited paper, repository, release note, pull request, or reproducible experiment.
- **Inferred**: a conclusion drawn from observed facts, but not directly stated by a source.
- **Hypothesized**: a claim that still requires experiments.

## 1. Scoped Position

- **Observed**: As of April 9, 2026, the anchor sources listed below do not clearly document an EAGLE-style drafter trained on hidden states or logits extracted from a quantized verifier.
- **Observed**: Public sources do document all of the adjacent pieces separately: draft training, speculative decoding with quantized verifiers, and mixed quantized-verifier plus unquantized-drafter inference.
- **Inferred**: There is a concrete gap between "quantized speculative inference is supported" and "quantized-teacher draft training is publicly documented."
- **Hypothesized**: Retraining a drafter against the quantized verifier can recover acceptance or latency lost from train/inference precision mismatch, without adding new inference-time stages or parameters.

This is the claim the paper should make in v1:

> Quantization-aware draft training is a testable way to recover speculative decoding performance on quantized verifiers, and current public evidence does not yet show that this training path has been documented end to end.

This is the claim the paper should **not** make yet:

> "The gap is genuinely open" after an exhaustive review of the literature.

That stronger claim should only appear after the draft includes a dated search protocol, explicit inclusion and exclusion criteria, and a source table that lets another reader reproduce the novelty audit.

## 2. Anchor Evidence Snapshot

The table below is a dated evidence snapshot, not a final literature appendix. A source counts as a positive example only if it explicitly states, or its training code clearly documents, that the drafter was trained on hidden states or logits from a quantized verifier.

| Source | Dated fact | What it directly supports | Trained on quantized teacher? |
| --- | --- | --- | --- |
| [P-EAGLE paper page](https://papers.cool/arxiv/2602.01469) | Published February 1, 2026 UTC on the linked paper page | Parallel-drafting EAGLE training exists at scale | **No public evidence in this source** |
| [SpecMQuant / CoRR abs/2505.22179](https://dblp.org/rec/journals/corr/abs-2505-22179.html) | 2025 CoRR record for the quantization plus speculative decoding paper | Quantized speculative decoding compatibility is a real problem; hierarchical inference is proposed | **No public evidence in this source** |
| [vLLM Speculators repository](https://github.com/vllm-project/speculators) | Accessed April 9, 2026 | Offline hidden-state generation, draft training, and deployment tooling exist in one stack | **Unclear in public docs; no explicit quantized-teacher example surfaced** |
| [vLLM PR #25883](https://github.com/vllm-project/vllm/pull/25883) | Merged September 29, 2025 | Quantized verifier plus unquantized drafter inference is a supported and tested scenario worth fixing | **No; inference-only bug fix** |

Working conclusion from the current evidence snapshot:

- **Observed**: The anchor sources support the existence of draft training infrastructure and quantized-verifier inference support.
- **Inferred**: The anchor sources do not yet provide a clean positive example of quantized-teacher draft training.
- **Inferred**: The novelty claim is currently best phrased as a provisional literature finding, not a final exhaustive-review statement.

## 3. Claims We Can Defend Now

- **Observed**: There is public tooling for hidden-state generation, drafter training, and vLLM deployment in the same ecosystem.
- **Observed**: Mixed precision deployments are real enough that vLLM needed a fix for Eagle3 quantization config inheritance when a quantized verifier was paired with an unquantized drafter.
- **Observed**: Quantization can interact poorly with speculative decoding at inference time, as documented by SpecMQuant.
- **Inferred**: Public evidence for quantized inference support does not imply public evidence for quantized-teacher training.
- **Inferred**: The current draft should frame the contribution as a fairness-tight empirical study plus a minimal training modification, not as a sweeping literature verdict.

## 4. Claims That Still Require Experiments

### H1. Precision mismatch creates a measurable performance gap

- **Hypothesized**: Under a matched training and inference stack, a BF16-trained control drafter loses mean accepted length and/or latency when evaluated against a W4A16 verifier instead of the BF16 verifier.
- **Working success threshold**: At least one primary metric worsens by 5% or more relative to the BF16-verifier control, with paired 95% confidence intervals excluding zero.

### H2. Quantization-aware training recovers some of the lost performance

- **Hypothesized**: A drafter trained against the W4A16 verifier outperforms the matched BF16-trained control on the same W4A16 verifier, with no extra inference-time stage.
- **Working success threshold**: The quantization-aware drafter recovers at least 25% of the acceptance loss, or improves end-to-end latency or ITL by at least 5%, relative to the matched BF16-trained control, while preserving exact output equivalence.

### H3. Recovery is consistent with reduced representation mismatch

- **Hypothesized**: Improvements in acceptance correlate with reduced hidden-state drift and improved logit agreement between the drafter-facing teacher signals and the quantized verifier.
- **Working success threshold**: At least one representation-level diagnostic moves in the same direction as the acceptance improvement, and the paper can show that verifier-side systems overhead alone does not explain the result.

These are proposal-stage thresholds. They should be revised only after the first pilot measurements are available.

## 5. Canonical Implementation Stack

Use one canonical stack for the main paper:

- `speculators + vLLM` for hidden-state generation, training, inference, and benchmarking.
- One model family first: Llama-3.1-8B or another single 8B-class family, but do not mix families in the core result.
- One quantization family first: W4A16 weight-only quantization.
- One drafter family first: EAGLE-3 style drafter.

Why this matters:

- **Inferred**: If the paper mixes `SpecForge`, `EAGLE`, and `speculators` in the main result, reviewers can argue that any effect comes from framework differences rather than teacher precision.
- **Observed**: `speculators` is already aligned with vLLM deployment and hidden-state generation, so it minimizes cross-stack confounds.

Contingency path:

- `SpecForge` can remain a backup implementation if `speculators` cannot expose the needed quantized-teacher training path, but that should be reported as contingency engineering, not as the main scientific setup.

### Low-Compute Gemma Pilot

- **Observed**: [vLLM's supported-models page](https://docs.vllm.ai/en/latest/models/supported_models/) lists `GemmaForCausalLM`, `Gemma2ForCausalLM`, and `Gemma3ForCausalLM` as supported text-generation architectures.
- **Observed**: [Google's Gemma documentation overview](https://ai.google.dev/gemma/docs) and [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) describe small Gemma 3 checkpoints, including 270M, 1B, and 4B variants, and position them for lower-resource deployments.
- **Observed**: The public `speculators` supported-model matrix currently lists Llama, Qwen3, and gpt-oss as end-to-end supported verifier families, but does not list Gemma.
- **Inferred**: Gemma is a good candidate for a cheap pilot that validates local model loading, chat-template rendering, hidden-state extraction, and small-scale draft-training plumbing.
- **Inferred**: A Gemma pilot should be treated as pipeline validation, not as the paper's primary empirical result, until the chosen `speculators + vLLM` stack is shown to support Gemma end to end for quantized-verifier training and evaluation.

Practical recommendation:

- Use `google/gemma-3-1b-it` for the first smoke tests when the goal is simply to verify that the local stack works.
- Use `google/gemma-3-4b-it` only if the 1B path succeeds and a slightly more realistic pilot is worth the extra cost.
- Do not use Gemma as the paper's anchor model family unless quantized verifier loading, teacher-signal extraction, drafter training, and evaluation all work cleanly in the canonical stack.
- Keep one publicly supported `speculators` verifier family, such as Llama or Qwen3, as the default paper target unless Gemma is validated end to end.

### Recommended Start Path

- **Inferred**: The safest start is a two-family strategy: `Gemma` for cheap local validation, then `Qwen3-8B` for the first paper-relevant pilot.
- **Inferred**: `Qwen3-8B` is a better first serious experiment target than Gemma because it is a publicly supported `speculators` verifier family and avoids the access friction of gated Llama checkpoints.
- **Inferred**: `Llama-3.1-8B-Instruct` should be treated as an optional follow-on target, not the first execution step, unless access is already in place and the team specifically wants Llama as the anchor family.
- **Inferred**: If Sol access at ASU is available, the project should no longer be planned around Colab-style constraints for the main pilot.

### Sol Compute Path

- **Observed**: The available Sol node types provided for planning include standard CPU nodes, high-memory CPU nodes, A30 GPU nodes, MIG slices from A100 GPUs, and `4x NVIDIA A100 (80 GiB)` GPU nodes with `512 GiB` host RAM.
- **Inferred**: The `4x A100 80 GiB` nodes are the preferred environment for the first paper-relevant pilot because they remove most of the memory pressure and session-fragility concerns that shaped the lower-compute fallback plan.
- **Inferred**: The `A30` nodes are acceptable for smoke tests, data preprocessing, or lighter evaluation, but they are a weaker default for the main matched-control training runs.
- **Inferred**: `MIG` slices are useful only if queue access is constrained; they are not the preferred environment for the main extraction and training path because they reduce per-job flexibility.

Practical compute recommendation:

- Use a local machine only for the first `Gemma 1B` smoke test.
- Use Sol `A100 80 GiB` nodes for the first real `Qwen3-8B` BF16 and AWQ extraction, evaluator validation, and matched-control training.
- Use Sol high-speed scratch storage for extracted hidden states, checkpoints, and benchmark logs rather than relying on home storage for large intermediate artifacts.
- Treat `Llama-3.1-8B-Instruct` as more realistic on Sol than on Colab because the A100 nodes remove most single-session memory pressure, but still keep Qwen3 as the first primary target unless there is a strong reason to prefer Llama.

Working plan:

#### Phase 0. Gemma local smoke test

- Run a local or CPU-bound smoke test with `google/gemma-3-1b-it`.
- Success criteria:
  - model download and tokenizer setup work
  - chat-template rendering works
  - hidden-state extraction works at the chosen tap layers
  - plain autoregressive generation returns a sensible short answer
- If this fails, fix the local environment before spending time on quantization or training.

#### Phase 1. Qwen3-8B stack validation

- Move immediately to `Qwen3-8B` for the first real experiment path.
- Run this phase on a Sol `A100 80 GiB` node if access is available.
- Validate the canonical stack on a tiny prompt set:
  - BF16 verifier load
  - prompt formatting and tokenizer stability
  - hidden-state extraction on a small JSONL slice
  - a short autoregressive benchmark pass
- Success criteria:
  - extracted tensor shapes and layer taps are stable across prompts
  - the same prompt file can be reused without tokenizer drift
  - the evaluator produces consistent exact-output checks on a mini benchmark

#### Phase 2. Qwen3-8B precision-mismatch pilot

- Use `Qwen3-8B` BF16 and AWQ checkpoints on the same prompts.
- Prefer Sol scratch storage for this phase so BF16 and AWQ extraction outputs can be regenerated or resumed without home-directory pressure.
- First measure verifier-side drift before full training:
  - hidden-state drift at drafter tap layers
  - logit agreement on the same prompts
  - any acceptance drop from a BF16-trained control drafter, if a small control drafter is already available
- Success criteria:
  - at least one mismatch diagnostic moves meaningfully between BF16 and AWQ
  - the quantized verifier path runs cleanly enough to justify matched training
- If there is no measurable mismatch signal, stop or narrow the project before larger training runs.

#### Phase 3. Qwen3-8B matched-control training

- Train exactly two fresh drafters on the same data and budget:
  - BF16-control drafter
  - W4A16 quantization-aware drafter
- On Sol, this phase should be treated as the default main run location rather than a fallback cluster option.
- Start with `1k` examples, then `8k`, and only then consider the larger run budget.
- Evaluate only the clean core rows first:
  - BF16-control -> BF16 verifier
  - BF16-control -> W4A16 verifier
  - Quantization-aware -> W4A16 verifier
- Success criteria:
  - the BF16-control degrades on W4A16
  - the quantization-aware drafter beats the matched BF16-control on the same W4A16 verifier

#### Phase 4. Optional Llama transfer

- Move to `Llama-3.1-8B-Instruct` only after the Qwen3-8B pilot shows a real signal.
- If Sol `A100 80 GiB` access is stable, this phase is operationally reasonable; if not, keep it out of scope.
- Use this phase only if one of the following is true:
  - Llama is strategically important for the paper
  - reviewers or collaborators will care more about Llama than Qwen3
  - the team already has stable access to the required checkpoints and quantized variants
- If Qwen3 does not show a clean effect, do not escalate to Llama just to rescue the story.

Decision rule:

- Start with `Gemma 1B` to de-risk the environment.
- Use `Qwen3-8B` on Sol for the first experiment that is meant to support the paper.
- Treat `Llama-3.1-8B-Instruct` as optional unless there is already a positive Qwen3 result or a strong external reason to prefer Llama.

## 6. Fairness-Tight Experimental Contract

The main comparison should use freshly trained matched controls rather than only public checkpoints.

### Required training runs

- Fresh BF16-control drafter trained with the exact same data, tokenizer, optimizer, sequence length, epoch budget, and compute budget as the proposed method.
- Fresh quantization-aware drafter trained against the W4A16 verifier under the same settings.
- Published public checkpoint used only as a secondary external baseline.

### Required evaluation configs

| Config | Drafter training target | Verifier weights | Verifier activations | KV cache | Drafter weights | Role |
| --- | --- | --- | --- | --- | --- | --- |
| AR BF16 baseline | None | BF16 | BF16 | Fixed across comparisons | N/A | Upper-bound verifier reference |
| AR W4A16 baseline | None | W4A16 | FP16 if weight-only quantization is used | Same as above | N/A | Quantized verifier reference |
| External public drafter -> W4A16 | Public checkpoint | W4A16 | FP16 if weight-only quantization is used | Same as above | FP16 or BF16 | Secondary external baseline |
| Matched BF16-control -> BF16 | BF16 | BF16 | BF16 | Same as above | FP16 or BF16 | Control |
| Matched BF16-control -> W4A16 | BF16 | W4A16 | FP16 if weight-only quantization is used | Same as above | FP16 or BF16 | Precision-mismatch test |
| Quantization-aware -> W4A16 | W4A16 | W4A16 | FP16 if weight-only quantization is used | Same as above | FP16 or BF16 | Proposed method |

Rules for keeping the comparison clean:

- Keep prompt formatting, tokenizer, decoding settings, speculative token budget, and benchmark prompts fixed.
- Keep KV-cache precision fixed across the primary comparison unless KV precision is itself the variable being studied.
- Hold drafter parameter count constant between the BF16-control and quantization-aware runs.
- Require speculative outputs to match autoregressive outputs from the same verifier.

## 7. Mechanism Diagnostics

Do not rely on end-to-end speed alone. The paper should include diagnostics that separate representation mismatch from plain verifier overhead.

### Required diagnostics

- Layerwise hidden-state drift between BF16 and W4A16 verifier runs at the teacher extraction points used for drafter training.
- Logit agreement metrics between BF16 and W4A16 verifier outputs on the same prompts.
- Acceptance by speculative depth, not just a single aggregate mean.
- Exact-output equivalence checks against autoregressive decoding on the same verifier.
- Latency decomposition into drafting time, verifier time, and end-to-end latency.

### Minimal interpretation rules

- **Observed**: If the W4A16 verifier is slower only because verification overhead dominates, that is a systems result, not a teacher-mismatch result.
- **Inferred**: A causal training-mismatch story is stronger only if the quantization-aware drafter improves acceptance on the same verifier and the representation diagnostics move in the same direction.

## 8. Evaluation and Reporting

### Primary metrics

- Mean accepted length
- Per-position acceptance rate
- Inter-token latency
- Time to first token
- End-to-end latency

### Statistical reporting

- Report paired prompt-level deltas, not just dataset averages.
- Use confidence intervals for all primary comparisons.
- Keep MT-Bench as one benchmark, but add one broader benchmark so the paper does not stand or fall on a single prompt set.

### Suggested benchmark posture

- Primary benchmark: MT-Bench
- One broader secondary benchmark: any benchmark with enough prompts to support paired confidence intervals and latency reporting

The core result should be framed as:

> Quantization-aware training improves speculative decoding on a quantized verifier relative to a matched BF16-trained control under the same stack and compute budget.

Not as:

> Quantization-aware training universally solves speculative decoding under quantization.

## 9. Kill Criteria and Pivot Paths

- If the matched BF16-control does **not** meaningfully degrade on W4A16, the core motivation is weak. Narrow the paper or stop.
- If the quantization-aware drafter does **not** beat the matched BF16-control on the same W4A16 verifier, the causal training story is weak. The result may still be publishable as a negative study, but not as a recovery claim.
- If the gain appears only in wall-clock speed and not in acceptance or representation diagnostics, reframe the result as a systems interaction paper rather than a training-alignment paper.
- Escalate to W4A8 or FP8 only if the W4A16 pilot suggests the effect exists but is too small to cleanly argue.

## 10. What This Draft Intentionally Removes from the Core Story

The following items were present in the earlier concept note but should stay out of the main paper draft until they are either measured or updated with exact dates:

- Predicted acceptance values and projected speedups
- Estimated training-hour claims not backed by pilot runs
- Colab compute-unit economics
- Venue recommendations and speculative submission deadlines

Those details can live in project notes. They should not carry the argument in the paper draft.

## 11. Deliverables for the Next Revision

The next paper revision should add these concrete artifacts:

- A novelty-audit appendix with dated search protocol, inclusion criteria, exclusion criteria, and source table
- A pilot results table for the matched BF16-control and quantization-aware W4A16 runs
- A mechanism figure showing hidden-state drift or logit agreement alongside acceptance changes
- A one-paragraph exactness statement confirming speculative outputs match the autoregressive verifier

## Anchor Sources

- [P-EAGLE paper page](https://papers.cool/arxiv/2602.01469)
- [SpecMQuant / CoRR abs/2505.22179](https://dblp.org/rec/journals/corr/abs-2505-22179.html)
- [vLLM Speculators repository](https://github.com/vllm-project/speculators)
- [vLLM PR #25883](https://github.com/vllm-project/vllm/pull/25883)
