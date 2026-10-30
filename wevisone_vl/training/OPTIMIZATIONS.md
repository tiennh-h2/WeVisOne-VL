# Training speed and memory improvements

The package keeps image resolution, all response tokens, source sampling,
LoRA settings, FP32 mask decoding and generation-based best-checkpoint selection.
No quantization or extra CUDA extension is required.

## Install

Extract `wevisone_vl_optimized.tar.gz` into the root of your existing WeVisOne-VL
repository. It contains the `wevisone_vl/` package, tests and this guide.
The training entry point and existing YAML files remain compatible:

```bash
tar -xzf wevisone_vl_optimized.tar.gz -C /path/to/WeVisOne-VL
python -m wevisone_vl.training --config configs/mixed_task_finetune.yaml
```

## Changes enabled automatically

- **Selective vocabulary logits:** compute the LM head and causal cross entropy
  only at hidden positions predicting a supervised assistant token. The exact
  next-token shift and ignored-label semantics are preserved. Prompt/image tokens
  still participate in transformer attention; their vocabulary logits are omitted.
  A backbone without an explicit `logits_to_keep` argument warns and falls back.
- **One image preprocessing pass:** reuse the complete conversation's Qwen grid
  to expand the prefix image token, then tokenize just the prefix text. The
  existing strict prefix-equality check still guards assistant-label alignment.
- **Frozen base dtype:** LoRA runs load frozen weights in BF16 or FP16 to match
  the requested mixed precision. Trainable parameters remain FP32. Previously
  the FP16 path loaded all frozen weights in FP32.
- **Smaller mask targets:** move binary targets as Boolean tensors (one byte per
  pixel versus four for FP32), converting inside the numerically stable loss.
  All target instances remain aligned and thin foreground remains preserved.
- **Detection-only inputs:** skip construction and transfer of the unused 1024px
  decoder canvas. Empty segmentation queries skip the canvas too.
- **Device-side metrics:** aggregate detached loss scalars on GPU rather than
  copying each scalar to CPU every microbatch. Nonfinite-loss checks remain.
- **Fused AdamW:** select the fused implementation when parameters are on CUDA
  and PyTorch exposes the option; use ordinary AdamW on CPU. Fused arithmetic
  may differ slightly due to floating-point ordering.
- **Generation replay:** segmentation validation projects only the last token's
  vocabulary logits during hidden-feature replay; predicted boxes/masks and
  quality-based checkpoint selection retain their previous meaning.

## Optional speed profile

Add this top-level section to your existing YAML:

```yaml
performance:
  selective_logits: true
  fast_prefix: true
  language_checkpointing: false
  decoder_checkpointing: false
  refinement_chunk_size: 0
  fused_adamw: auto
  prefetch_batches: 2
```

A single background CPU worker prepares a bounded number of upcoming batches.
This overlaps file reading, mask decoding, image processing and tokenization with
GPU execution. It uses the original deterministic source/index schedule, including
on resume, and never samples masks or touches CUDA. More prefetch consumes more
host RAM; start at 1 or 2. Default is 0 (synchronous input preparation).
The processor is locked during generation validation to avoid concurrent tokenizer
state mutation. PyTorch mask resizing can use CPU threads; avoid oversubscribing
cores when running several training ranks.

## Lower-VRAM profile

Use the speed profile above and set:

```yaml
performance:
  selective_logits: true
  fast_prefix: true
  language_checkpointing: true
  decoder_checkpointing: true
  refinement_chunk_size: 2
  fused_adamw: auto
  prefetch_batches: 2
```

- Language checkpointing recomputes only LM blocks during backward, using
  `use_reentrant=False`. This supports frozen embeddings and differentiable
  last-layer feature capture. Vision checkpointing stays disabled, including with
  vision LoRA, to avoid changing the pre-merger feature-capture path.
- Decoder checkpointing recomputes the decoder **and mask loss**, saving the image,
  captured features, selected queries, boxes and targets instead of large internal
  FP32 refinement activations. Decoder weights and execution remain FP32.
- `refinement_chunk_size: 2` bounds mask prediction, full-canvas refinement,
  and mask loss to two selected objects per chunk. Use `1` if still memory-limited;
  `0` (the default) retains the earlier unchunked execution. Joint query attention
  still runs over every selected object before chunks are formed. The original
  decoder configuration, 1024-pixel canvas (for the released model), FP32
  execution, parameter names/shapes, and strict pretrained weight loading remain
  unchanged. No selected mask is dropped by chunking. Per-chunk means are
  weighted by actual chunk sizes, including a shorter final chunk.
- Combine a positive chunk size with `decoder_checkpointing: true`: shared image
  preparation and each refinement/loss chunk are checkpointed separately. Without
  checkpointing, autograd retains activations across chunks, limiting savings.
  `language_checkpointing` remains independent. Inference uses its existing path;
  this training option does not alter generation validation batching.
- Checkpointing trades compute for activation memory. It may slow each step;
  use it when memory limits your image/token budget. No fixed percentage of
  speedup or VRAM reduction has been measured on a production GPU here.

For maximum compatibility, set `selective_logits: false`, `fast_prefix: false`,
`language_checkpointing: false`, `decoder_checkpointing: false`,
`refinement_chunk_size: 0`,
`fused_adamw: false`, and `prefetch_batches: 0`.

## DDP and resume

`find_unused_parameters=True` remains necessary for mixed workloads: detection
queries skip mask-decoder parameters, and negative segmentation queries may do so
as well. Disabling it unconditionally can break subsequent DDP iterations.

Performance options do not change parameter names/shapes or optimizer group
ordering, and are excluded from the resume signature. Existing checkpoints remain
loadable with the original training/data settings and world size. Mathematical
loss/gradient equivalence does not promise bitwise-identical CUDA trajectories
when kernels or numerical execution order change. Keep all performance settings
unchanged for strict reproducibility. Export format stays compatible.

## Measure on your GPU

Training events in `metrics.jsonl` now include `performance.seconds_per_step`,
`queries_per_second`, `peak_allocated_gib` and `peak_reserved_gib` (CUDA only).
Timing covers optimizer-update attempts, including all accumulation microbatches,
and excludes generation-validation and save time. Throughput is global query
count divided by this rank's elapsed time; it is an estimate, not a distributed
benchmark. CUDA peaks include model/optimizer storage and are reset after
validation to avoid counting generation cache peaks. Startup steps include
optimizer allocation and warmup; compare steady-state windows with identical
input budgets, seeds, accumulation, world size and validation settings.

Use a separate output directory for each benchmark. Start with the automatic
optimizations, then try prefetching, then checkpointing if VRAM is tight. Keep
`max_pixels`, `max_masks_per_sample`, LR and LoRA rank unchanged when comparing.

## Verification

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests
python -m compileall -q wevisone_vl
```

Tests use tiny random Qwen3-VL models, real PEFT language/vision adapters and the
actual pixel decoder, without downloading weights. They compare eager/full-logit
losses and parameter gradients with selective logits and optional checkpointing,
including frozen-vision cases. Real Qwen processor tests check the prefix path.
A training-loop integration test checks accumulated optimizer updates, prefetched
inputs, checkpoint save and exact resume.
The uploaded archive did not contain its earlier project's test suite. Full-size
GPU speed, memory, task accuracy and multi-GPU behavior require validation on your
training machine.
