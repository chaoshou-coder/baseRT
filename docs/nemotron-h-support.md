# Supporting NVIDIA Nemotron-H in baseRT — Feasibility Analysis & Evidence Pack

> **Date**: 2026-08-12 · **Status**: draft, prepared for upstream review
>
> **Target model**: [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16) (HF safetensors, 14 shards)
> **Codebase**: baseRT v0.2.0 (local checkout `v0.2.0-30-g9116053`; converter open-source, runtime closed-source)
> **Purpose**: submission material for a feature request — architecture profile, tensor inventory, converter gap list (with file:line refs), runtime requirements, and a ready-to-paste issue draft (§7)

---

## TL;DR

- Nemotron-H is a **Mamba-2 + MoE + full-attention hybrid** (52 layers = 23 SSM + 23 MoE + 6 attention). A converter-side mapper is writable, but **no existing runtime class can execute it** — the runtime (closed-source) has no Mamba-2 kernels, and GDN kernels compute different math, so name-remapping tricks are not an option.
- **Why it's worth it**: NVIDIA ships official **D-Flash / D-Spark speculative-decoding variants** for this exact model, and MoE is the productive shape for agent/harness workloads (7.3× prefill vs dense, measured) — baseRT currently has no speculative-decoding path at all.
- **What we're asking for**: a new `nemotron_h` model class in the runtime + a converter mapper. **We can contribute the converter side** (§5 lists exactly what it takes, line-by-line); the runtime class is the prerequisite (§6).

---

## 1. Executive Summary

1. **A converter-side `nemotron_h` mapper is technically writable, but it cannot make the model run on baseRT.** The blocker is not the converter — it is the (closed-source) runtime: no existing runtime model class can execute the Nemotron-H architecture (Mamba-2 SSM + MoE + full-attention hybrid), and none of the existing classes combine Mamba-2 kernels with MoE:
   - `qwen3_moe` class: standard full-attention MoE, **no SSM kernels at all**;
   - `qwen35moe` class: GDN (Gated DeltaNet) + MoE — GDN is a *different* linear-attention family; its kernels compute different math than Mamba-2, so remapping tensor names onto GDN names cannot work (the kernels would produce garbage);
   - `llama` class: no MoE.
2. **The "cheapest" option — transcoding Nemotron onto the `qwen3_moe` runtime class — is not viable**: the 23 mamba blocks have no target kernels to execute. Load fails on missing tensors, or the output is wrong.
3. **Motivation goes beyond the base architecture** (§2): NVIDIA officially ships **D-Flash and D-Spark speculative-decoding variants for this exact model**, and MoE is the best shape for agent/harness workloads (prefill-dominated, 7.3× measured vs dense). baseRT currently has no speculative-decoding path at all (MTP is dropped at convert time, `main.rs` ~1988-1990; decode is memory-bandwidth-bound). Supporting Nemotron-H makes baseRT the platform for this optimization class on Apple Silicon.
4. **Recommended path**: submit an architecture-support request to basecompute/baseRT (runtime + converter sides). llama.cpp already has a complete reference implementation (`LLM_ARCH_NEMOTRON_H` — commit [cce8cb1](https://github.com/ggml-org/llama.cpp/commit/cce8cb1bc55eafda3830c62ee5ae2b12ea6b80dd), [PR #15572](https://github.com/ggml-org/llama.cpp/pull/15572), conversion fixes in [PR #21664](https://github.com/ggml-org/llama.cpp/pull/21664)). A ready-to-paste issue draft is in §7.

---

## 2. Motivation

### 2.1 Beyond MTP: NVIDIA's official D-Flash / D-Spark variants

Nemotron-3.5-Lightning-30B-A3B is not just another MoE — it is the reference platform for NVIDIA's current **speculative-decoding** stack:

- **Official variant repos exist for this exact model** (nvidia org on HF):
  - [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash)
  - [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark)
  - Each repo ships a **standalone draft model** (verified: `model_type: qwen3`, `architectures: [DFlashDraftModel]`) with spec-decode config keys (`dflash_config`, `dflash_query_causal`, `dspark_fixup_head_type`, `dspark_markov_rank`, `eagle_aux_hidden_state_layer_ids`, `num_aux_layers`, `sample_from_anchor`, `mask_token_id`, `pard_token`, `draft_vocab_size`, …). The draft pairs with the target checkpoint for block-wise speculative decoding.
- **D-Flash** ([vLLM speculators docs](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md)): a small diffusion-LLM draft model predicts a whole block of tokens in a single forward pass — non-causal (bidirectional) attention over the verifier's hidden states + anchor-point block drafting; block-generation cost is O(1) regardless of block size. Reported gains: **2-3× larger speedups than Eagle-3** on synchronous requests, **6.08× peak single-stream speedup** on MATH-500 (Qwen3-8B), **15× production throughput** on Blackwell (gpt-oss-120b).
- **D-Spark**: D-Flash's parallel backbone plus a **sequential Markov head** (stable acceptance at depth) and a **confidence head** (dynamic, load-aware verification). Reported: **60-85% production throughput gain** in a live DeepSeek-V4 deployment ([vLLM PR #46995](https://github.com/vllm-project/vllm/pull/46995)).

### 2.2 Why MoE wins for agent / harness workloads

Agent and harness workloads (tool-calling loops, RAG, multi-turn reasoning — the dominant LLM usage pattern) are **input-dominated**: every turn re-processes the system prompt, tool results and conversation history, so prefill tokens far outnumber generation tokens (a 2k-token answer over a 32k context is ~94% prefill). The experience-impacting factors, in order of importance:

1. **Prefill throughput** — time-to-first-token and context re-processing dominate perceived latency;
2. **Parallelism / concurrency** — many in-flight requests (batched agents, parallel sub-tasks) with tight P95;
3. **Decode speed** — single-stream tokens/s matters least (bandwidth-bound in every engine).

MoE is the best shape for this profile: prefill FLOPs scale with **active** parameters (A3B ≈ 3B active), so prefill throughput reaches levels dense models of the same total size cannot approach — while routed-expert capacity keeps the intelligence trade-off minimal (the A3B design exists precisely to preserve quality at low active cost). Measured on the same M5 Pro, same harness, Q4:

| Model | prefill pp2048 | decode tg128 |
|---|---|---|
| Qwen-AgentWorld-35B-A3B (MoE) | 3028 t/s | 110 t/s |
| Qwen-27B-DavidAU (dense) | 417 t/s | 17 t/s |
| **ratio** | **7.3×** | 6.5× |

A 7.3× prefill gap is exactly the experience difference agent workloads feel — and it grows with context length. Nemotron-3.5-Lightning-30B-A3B is the same A3B shape (30B total / 3B active) from NVIDIA's flagship line, with MTP and D-Flash/D-Spark on top: the best-case package for this workload class.

### 2.3 What this means for baseRT

Today baseRT has no speculative decoding: MTP weights are dropped at convert time (qwen35 gate in `to_canonical_name`, `main.rs` ~1988-1990), and decode is bandwidth-bound (~110 t/s on M5 Pro). Adding Nemotron-H — keeping MTP, or loading the D-Flash draft model — is the prerequisite for bringing this optimization class to Apple Silicon.

A useful side note: the draft model itself is `model_type: qwen3`, which baseRT's existing `QwenHfMapper` could already convert; only the runtime speculative-decoding path is missing.

---

## 3. Architecture Profile

Source: `config.json` (top level of the HF repo; `model_type: "nemotron_h"`, `architectures: ["NemotronHForCausalLM"]`).

### 3.1 Overview

| Field | Value |
|---|---|
| model_type / architectures | `nemotron_h` / `NemotronHForCausalLM` |
| Size | 30B total / 3B active (A3B, per model card) |
| Layers | 52 (`num_hidden_layers`) |
| hidden_size | 2688 |
| Attention | 32 heads / 2 KV heads, head_dim 128, `partial_rotary_factor: 1.0`, `rope_theta: 10000`, **no q_norm/k_norm** |
| vocab_size | 131072 (padded), `tie_word_embeddings: false` |
| Context | 262144 (extensible to ~1M per model card) |
| Compute dtype | bfloat16 |
| Norm eps | `layer_norm_epsilon: 1e-05` (note: differs from baseRT's 1e-6 default) |

### 3.2 Block schedule (`layers_block_type`, 52 entries)

**23 × `mamba` + 23 × `moe` + 6 × `attention`**, interleaved; attention at layers 5/12/19/26/33/42 (0-indexed). Each block is preceded by a per-layer norm (`backbone.layers.N.norm.weight`):

```
layers   0  1  2  3  4  5   6 ... 42  43 ... 51
type     M  E  M  E  M  A   E ...  A   E  ... E     M=mamba  E=moe  A=attention
```

### 3.3 Mamba-2 block parameters

`ssm_state_size: 128`, `conv_kernel: 4`, `chunk_size: 128`, `expand: 2`, `mamba_head_dim: 64`, `mamba_num_heads: 64`, `mamba_hidden_act: silu`, `mamba_ssm_cache_dtype: float32`, `time_step_min/max/floor: 0.001/0.1/0.0001`, `use_mamba_kernels: true`.

### 3.4 MoE block parameters (DeepSeek-style grouped experts)

| Field | Value |
|---|---|
| `n_routed_experts` | **128** (note: not the `num_experts` key!) |
| `num_experts_per_tok` | 6 |
| `n_groups` / `topk_group` / `n_group` | 8 / 1 / 1 (grouped top-k) |
| `moe_intermediate_size` | 1856 |
| `moe_shared_expert_intermediate_size` | 3712 |
| `n_shared_experts` | 1 (`moe_shared_expert_overlap: true`) |
| `routed_scaling_factor` | 2.5 |
| `norm_topk_prob` | true |
| **`mlp_hidden_act`** | **`relu2`** (not SiLU) |
| `moe_latent_size` | null (no latent projection) |

Consequence of `relu2`: experts need **no gate projection** — each expert is exactly two matrices (`up_proj` + `down_proj`), confirmed by the tensor inventory (§4.3).

### 3.5 MTP (present, 2 layers)

`mtp_layers_block_type: ["attention", "moe"]`, `num_nextn_predict_layers: 1`. Tensors under `mtp.layers.*` (264 tensors).

---

## 4. Tensor Naming Inventory

Source: `model.safetensors.index.json` (6513 tensors). **The root prefix is `backbone.` — there is no `model.` prefix**, which misaligns with the prefix-strip chain in baseRT's `to_canonical_name` (§5.3).

### 4.1 Top level

```
backbone.embeddings.weight          # token embeddings [131072, 2688]
backbone.norm_f.weight              # final norm (note: NOT named norm.weight)
lm_head.weight                      # [131072, 2688]
```

### 4.2 Mamba blocks (23 layers, 8 tensors each)

```
backbone.layers.N.norm.weight             # pre-block norm (all 52 layers) [2688]
backbone.layers.N.mixer.A_log             # bare param (no .weight) shape [64] BF16 — state-decay matrix!
backbone.layers.N.mixer.D                 # bare param shape [64] — state-injection matrix
backbone.layers.N.mixer.dt_bias           # bare param shape [64] — time-step bias
backbone.layers.N.mixer.conv1d.weight     # [6144, 1, 4] 3-D (4-wide last dim cannot form a sane quant group)
backbone.layers.N.mixer.conv1d.bias
backbone.layers.N.mixer.in_proj.weight    # [10304, 2688] fused input projection (x+z gating)
backbone.layers.N.mixer.out_proj.weight   # [2688, 4096]
```

**Important**: `A_log`/`D`/`dt_bias` share names with baseRT's GDN tensors (`linear_attn.A_log`/`dt_bias`) but live at a **different path** (`mixer.` vs `linear_attn.`), and `D` does not exist in GDN at all. All three are 1-D [64], which drops them straight into baseRT's "1-D ⇒ norm-like ⇒ forced f16" rule (`main.rs:1509`) — see §5.3.

### 4.3 MoE blocks (23 layers)

```
backbone.layers.N.mixer.gate.weight                        # router [128, 2688]
backbone.layers.N.mixer.gate.e_score_correction_bias       # bare param — DeepSeek-style score correction
backbone.layers.N.mixer.experts.{0..127}.up_proj.weight    # [1856, 2688]
backbone.layers.N.mixer.experts.{0..127}.down_proj.weight  # [2688, 1856]
backbone.layers.N.mixer.shared_experts.up_proj.weight      # [3712, 2688]
backbone.layers.N.mixer.shared_experts.down_proj.weight    # [2688, 3712]
```

Key differences from every MoE baseRT already supports:

- **No `gate_proj` per expert** (relu2 activation) — baseRT's expert naming/StackingProvider expects the `{gate,up,down}_proj` triple;
- The router is `mixer.gate.weight`, not `mlp.gate.weight` / `mlp.router.weight` — existing profile rules `**.router.weight` / `**.mlp.gate.weight` do not match, and the `**.weight` catch-all would **silently quantize it to Q4** (defeating the "keep router high-precision" intent);
- `e_score_correction_bias` is a new tensor concept baseRT has no counterpart for.

### 4.4 Attention blocks (6 layers)

```
backbone.layers.N.mixer.{q,k,v,o}_proj.weight    # standard GQA, no q_norm/k_norm
```

### 4.5 MTP (2 layers, 264 tensors)

```
mtp.layers.0.{norm,hnorm,enorm}.weight  +  mtp.layers.0.eh_proj.weight
mtp.layers.0.mixer.{q,k,v,o}_proj.weight                          # attention block
mtp.layers.1.mixer.experts.{0..127}.{up,down}_proj.weight         # moe block
mtp.layers.1.mixer.gate.weight / e_score_correction_bias
mtp.layers.1.mixer.shared_experts.{up,down}_proj.weight
mtp.layers.1.final_layernorm.weight
```

---

## 5. Converter Gap List (file-precise)

Converter source: local `/Users/bang/baseRT/base-convert/`. This is exactly what a `nemotron_h` mapper requires, each item annotated with its current-state location. Line numbers refer to the local checkout.

### 5.1 Registration (`base-arch/src/lib.rs`)

- Add a `"nemotron_h"` arm to `hf_mapper_for_model_type` (`lib.rs:77`);
- Add an entry to `SUPPORTED_HF_MODEL_TYPES` (`lib.rs:129`) — **must stay in lockstep with the match**, or the `supported_list_matches_dispatch` test (`lib.rs:482-486`) fails;
- New module `base-arch/src/nemotron.rs`;
- GGUF side `source_mapper_for_gguf` (`lib.rs:32`): if llama.cpp exports `nemotron_h` GGUFs, add an arm there too (currently the dispatch table covers only llama/qwen family/gemma/nomic-bert).

**Verified today**: `./basert convert` on a nemotron_h config fails with `HF model_type "nemotron_h" not supported yet` (mapper lookup inside `convert_hf`, `main.rs:759`).

### 5.2 New `HfMapper` implementation (`nemotron.rs`)

The `HfMapper` trait (`lib.rs:47-75`) has four methods: `canonical_arch()`, `config_from_hf()`, `norm_shift()` (default 0), `rope_permute_heads()` (default None).

- **`canonical_arch()` must honestly return `"nemotron_h"`.** Transcoding onto `qwen3_moe` (the QwenMoeHfMapper precedent, `qwen.rs:179`) is mathematically invalid here (§1); the phi3/mistral → `"llama"` precedent (comment at `lib.rs:79-97`) holds only when the architectures are fully compatible.
- **`config_from_hf` cannot reuse `hf_generic_config` (`llama.rs:47`) directly** — it hard-requires `hidden_size`/`num_hidden_layers`/`num_attention_heads`/`intermediate_size`/`vocab_size`. Nemotron's MoE fields are `n_routed_experts` (≠ `num_experts`), `n_shared_experts`, `moe_intermediate_size`, `routed_scaling_factor`, `mlp_hidden_act=relu2`, `norm_topk_prob=true`, plus `layer_norm_epsilon=1e-5` (≠ baseRT default 1e-6). Follow the MoE-reading pattern of QwenMoeHfMapper (`qwen.rs:179-207`) and the `intermediate_size` backfill trick of Qwen35MoeHfMapper (`qwen.rs:121-141` — patch `intermediate_size` from `moe_shared_expert_intermediate_size` when absent).
- **`ArchConfig` (`lib.rs:170-302`) lacks fields**: `routed_scaling_factor`, `n_groups`/`topk_group`, `mlp_hidden_act`, and the full `mamba_*` set — the struct and `to_config_map()` (`lib.rs:307-475`) must be extended so the header carries them.
- **`rope_permute_heads`**: determine whether Nemotron q/k_proj use the HF rotate_half layout; llama/mistral/phi3 return head counts (`llama.rs:35`), and skipping the permutation collapses long-context retrieval (comment at `lib.rs:61-71`).
- **`norm_shift`**: confirm whether Nemotron norms are zero-centered (1-D norms are forced to f16 anyway per §5.4, but the +1 offset semantics must be checked).

### 5.3 `to_canonical_name` (`base-convert/src/main.rs:1982`)

Current strip chain (`main.rs` ~2043-2048): `model.language_model.` → `language_model.model.` → `language_model.` → `model.`. **Nemotron's `backbone.` prefix is not in the chain** — top-level tensors would keep names like `backbone.embeddings.weight` (the runtime looks up `embed_tokens.weight` → FATAL).

Required handling per tensor pattern:

| Pattern | Current behavior | Needed |
|---|---|---|
| `backbone.embeddings.weight` | matches no strip/rename | → `embed_tokens.weight` |
| `backbone.norm_f.weight` | the `norm.weight → final_norm.weight` rename (`main.rs` ~2060-2073) only fires on stripped `norm.weight` | → `final_norm.weight` |
| `backbone.layers.N.norm.weight` | `input_layernorm → input_norm` chain (`main.rs` ~2165) does not cover `layers.N.norm` | → `layers.N.input_norm.weight` (or per runtime alias convention) |
| `backbone.layers.N.mixer.experts.N.{up,down}_proj.weight` | the `.mlp.experts.`/`.experts.` replace chain (`main.rs` ~2093-2145) would mis-map names through `mixer.` (e.g. `mixer.ffn_gate_up_exps.weight`), which the runtime alias table does not resolve → **FATAL at load** | new `mixer.experts.N.{up,down}_proj` → runtime-canonical expert names (`ffn_{up,down}_exps.weight`); **and no `gate_proj`** — decide how the runtime handles two-matrix experts |
| `backbone.layers.N.mixer.gate.weight` | no rule matches; the `**.weight` catch-all silently quantizes the router to Q4 | map to the router canonical (`mlp.router.weight` / `ffn_gate_inp.weight`) |
| `mixer.gate.e_score_correction_bias` | no counterpart exists | new canonical name + runtime support |
| `mixer.A_log` / `D` / `dt_bias` (all 1-D [64]) | SSM predicates miss: `is_gdn_a_log` (`main.rs:1957`) recognizes only `.linear_attn.A_log`; `is_ssm_a` only `.ssm.a_log`/`.ssm_a` — so these fall into `is_norm_like = shape.len()==1 \|\| is_gdn_f16(...)` (`main.rs:1509`) → **forced f16**, a NaN disaster for Mamba state matrices (see the GDN comment near `main.rs:1950`); `D` has no counterpart either (`ssm.d` check matches only `.ssm.d`) | extend the SSM predicates + `SSM_A_MATRIX` f32/CPU mechanism (`main.rs` ~1500) to cover `mixer.{A_log,D,dt_bias}` |
| `mixer.conv1d` (3-D [6144,1,4]) | `is_gdn_f16` (`main.rs:1972`) recognizes only `linear_attn.conv1d/in_proj_a/in_proj_b`; a 3-D tensor caught by the `**.weight` catch-all would be packed as base_q4 — the 4-wide last dim cannot form a sane quant group (GDN comment) | own f16 protection for the mamba conv1d |
| `mixer.in_proj` / `out_proj` (2-D linears) | `**.weight` → base_q4 is structurally fine | optional f16 protection mirroring GDN, otherwise Q4 is usable |

### 5.4 Quant decision (`main.rs convert_generic`, 1264)

- `has_moe` heuristic (`main.rs` ~1440-1442): source names containing `experts` — `mixer.experts` matches ✓ (and `shared_experts` also contains `experts` ✓);
- `has_ssm` / `has_hybrid` heuristics (same region): keyed on the `ssm_` prefix — Nemotron source names are `mixer.A_log` etc., no `ssm_` → **flag not set**, needs extension;
- 1-D forced-f16 (`main.rs:1509`): `A_log`/`D`/`dt_bias` land here unless the SSM predicate claims them first (§5.3);
- **Two-matrix experts (up/down only)**: StackingProvider (`main.rs` ~1314, stacks per-expert matrices into 3D) and the runtime kernels must accept gate-less experts (relu2 semantics).

### 5.5 Profile (`base-quant/src/profile.rs`)

- Arch validation: **currently not enforced** — `validate()` (`profile.rs:91`) only requires a non-empty `arch` (the doc comment at `profile.rs:28` claims "mismatch is an error" but no implementation exists). Both `"arch": "*"` (used by the official default-q4/q8 profiles) and `"nemotron_h"` work;
- The rule layer of the existing q4mix profile (Linear Q4 / norm / router / lm_head globs) is reusable, but the router rule must target the new canonical name (`**.gate.weight` or the mapped router name).

---

## 6. Runtime Requirements (for the baseRT author)

The (closed-source) runtime needs a new `nemotron_h` model class with at least:

1. **Mamba-2 SSM block forward**: conv1d + state recurrence + time-step gating + x/z gating (fused `in_proj`), chunked computation at `chunk_size=128`; recurrent-state cache for decode (follow llama.cpp's practice of allocating recurrent cache only for SSM layers via `is_recurrent()`);
2. **Grouped top-k MoE**: pick 1 of 8 groups (`n_groups=8, topk_group=1`) + 6 experts within the group, `routed_scaling_factor=2.5` scaling, `norm_topk_prob` normalization, `e_score_correction_bias` score correction;
3. **relu2-activated experts**: up → relu2 → down, no gate projection;
4. **Shared expert** (up/down pair, 3712 wide);
5. **MTP / speculative-decoding decision**: keep MTP, or load the official D-Flash/D-Spark draft model (§2.1) — this is where the end-to-end speedups come from;
6. **Mixed block schedule**: `layer_types` per layer + per-layer pre-norm.

**Reference implementation**: llama.cpp `LLM_ARCH_NEMOTRON_H` (commit [cce8cb1](https://github.com/ggml-org/llama.cpp/commit/cce8cb1bc55eafda3830c62ee5ae2b12ea6b80dd) + [PR #15572](https://github.com/ggml-org/llama.cpp/pull/15572); GGUF `layer_types` 0=SSM/1=ATTN/2=FFN; conversion-script fixes in [PR #21664](https://github.com/ggml-org/llama.cpp/pull/21664)).

---

## 7. Upstream Issue Draft (ready to paste)

**Title**: Support NVIDIA Nemotron-H architecture (Mamba-2 + MoE + Attention hybrid, e.g. Nemotron-3.5-Lightning-30B-A3B)

**Body**:

```
Requesting converter + runtime support for the Nemotron-H architecture
(`model_type: nemotron_h`), e.g. nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16.

Architecture (config.json):
- 52 layers: 23× Mamba-2 SSM + 23× MoE + 6× full attention (layers_block_type)
- MoE: 128 routed experts (n_routed_experts), 6 per token, grouped top-k
  (n_groups=8, topk_group=1), routed_scaling_factor=2.5, norm_topk_prob,
  1 shared expert (3712 wide), mlp_hidden_act=relu2 (experts have no gate_proj)
- Mamba-2: ssm_state_size=128, conv_kernel=4, chunk_size=128
- MTP present (mtp_layers_block_type=[attention, moe], 2 layers)
- 32 attn heads / 2 KV, head_dim 128, full RoPE

Why it matters: beyond MTP, NVIDIA ships official D-Flash and D-Spark
speculative-decoding variants for this exact model
(nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash / -DSpark,
standalone qwen3-arch draft models). D-Flash block-parallel drafting
reports 2-3x speedups over Eagle-3 and up to 6.08x peak single-stream
(MATH-500) / 15x production throughput (Blackwell); D-Spark adds a
sequential Markov head + confidence head and reports 60-85% production
throughput gains (live DeepSeek-V4). baseRT currently has no
speculative-decoding path (MTP is dropped at convert; decode is
bandwidth-bound), so Nemotron-H support is the prerequisite for this
optimization class on Apple Silicon.

For agent/harness workloads (input-dominated, prefill-heavy), MoE is the
productive shape: measured on M5 Pro with Q4, a 35B-A3B MoE prefills at
3028 t/s vs 417 t/s for a 27B dense (~7x), with minimal intelligence
trade-off. For this usage class, prefill throughput and concurrency
matter more than decode — and MoE wins decisively on both.

Current state: base-convert rejects it at the mapper stage
(`HF model_type "nemotron_h" not supported yet`); the converter's
to_canonical_name prefix chain (model.language_model./model.) does not
cover the `backbone.layers.N.mixer.*` naming, MoE experts are up/down-only,
the SSM tensors (mixer.A_log/D/dt_bias) fall outside the existing
GDN/ssm f32-protection predicates and would be forced to f16 by the
1-D norm-like rule, and no runtime model class can execute Mamba-2 blocks.

Full evidence pack (tensor inventory with shapes, converter gap list with
line refs, runtime requirements): docs/nemotron-h-support.md (attached).

llama.cpp already ships a reference implementation
(LLM_ARCH_NEMOTRON_H, GGUF layer_types 0=SSM/1=ATTN/2=FFN).
Happy to contribute the converter-side mapper once runtime support is planned.
```

---

## Appendix A: Measured Evidence

Probe with the real `config.json` + an empty safetensors shard:

```
base-convert v0.2.0
  input:   /tmp/nemotron-probe
  output:  /tmp/nemotron-probe/out.base
  profile: qwen35-35b-a3b-q4mix
  arch:    nemotron_h
Error: HF model_type "nemotron_h" not supported yet
```

## Appendix B: Source Index

| Evidence | Origin |
|---|---|
| config.json fields | [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16/config.json`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16/blob/main/config.json) |
| Tensor names & shapes | same repo `model.safetensors.index.json` (6513 tensors); shapes read from shard-1 header (range request) |
| D-Flash/D-Spark variants | [`...-NVFP4-DFlash`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash) / [`...-NVFP4-DSpark`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark) (config `model_type: qwen3`, `DFlashDraftModel`; draft config keys `dflash_config`, `dspark_markov_rank`, `sample_from_anchor`, …) |
| D-Flash/D-Spark speedup figures | vLLM speculators docs ([`docs/user_guide/algorithms/dflash.md`](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md)); [vLLM PR #46995](https://github.com/vllm-project/vllm/pull/46995) |
| Converter line references | local `/Users/bang/baseRT/base-convert/` (v0.2.0-30-g9116053) |
| Runtime arch set | `include/baseRT/types.h` (llama/qwen3/gemma/gemma3/gemma4/whisper — no hybrid-MoE class) |
| llama.cpp reference | commit [cce8cb1](https://github.com/ggml-org/llama.cpp/commit/cce8cb1bc55eafda3830c62ee5ae2b12ea6b80dd), [PR #15572](https://github.com/ggml-org/llama.cpp/pull/15572), [PR #21664](https://github.com/ggml-org/llama.cpp/pull/21664) |
