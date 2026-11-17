# DeepInfer Unified CED-PD

A code-review prompt for evaluating a unified CED-PD execution path in a customized DeepInfer runtime.

## Contents

- `prompts/deepinfer_unified_ced_review.md` — the full Chinese, instruction-oriented prompt intended to be supplied to DeepSeek together with access to the local source repositories.

## Intended workflow

1. Clone this repository on the machine where DeepSeek can inspect the code:
   ```bash
   git clone https://github.com/ml1019-lmx/deepinfer-unified-ced.git
   cd deepinfer-unified-ced
   ```
2. Open `prompts/deepinfer_unified_ced_review.md` and provide its full contents to DeepSeek.
3. Make sure DeepSeek can read the local checkout of your **improved DeepInfer** and the relevant `vllm-ascend` checkout. The improved DeepInfer checkout is the primary implementation baseline.
4. Ask DeepSeek to perform the repository audit and produce a file/function-level implementation plan. The prompt asks for analysis and a plan; it does not authorize automatic source modifications.

## Reference repositories and report

- [DeepSeek-AI/DeepInfer](https://github.com/deepseek-ai/DeepInfer) — official upstream reference.
- [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) — Ascend/vLLM execution, scheduling, model-runner and KV-transfer reference.
- [CED-PD performance report (2026-09-25)](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/main/docs/CED-PD-PERF-20260925.md) — experimental CED-PD design and measured results.

## Scope

The prompt deliberately excludes basic environment setup, hardware adaptation, dependency compilation, and CANN/backend initialization. It focuses on CED layer execution, KV-group transfer, D-side replay, unified scheduling, unified ModelRunner behavior, CUDA Graph integration, correctness, and performance validation.

The public DeepInfer README states that a KV-cache system and VE/Prefill/Decode disaggregation are not included in that upstream release. Therefore, the local improved DeepInfer checkout—not assumptions about upstream feature completeness—must be treated as the source of truth for the implementation review.
