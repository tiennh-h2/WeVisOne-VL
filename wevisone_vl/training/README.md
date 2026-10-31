# Mixed detection and instance-segmentation fine-tuning

For the new full-parameter Rex-style SFT/FSDP1 recipe, see
[REX_SFT_GUIDE.md](../../REX_SFT_GUIDE.md) and `configs/rex_sft.yaml`.
The legacy training path below remains the default unless `recipe: rex_sft` is selected.

This extension fine-tunes a released Tencent WeVisOne-VL checkpoint using mixed
COCO sources. It reuses the original architecture and leaves existing inference
code unchanged. Upstream baseline: `40f735b8daed437ece982db9718e9c9856f52a79`.


## Install and configure

Replace `wevisone_vl/training/` in your existing WeVisOne-VL clone with this
folder. This archive contains `training/` at its root:

```bash
tar -xzf training.tar.gz -C /path/to/WeVisOne-VL/wevisone_vl
```

Install the normal upstream dependencies plus the training dependencies:

```bash
pip install accelerate peft pycocotools pyyaml
```

Edit `configs/mixed_task_finetune.yaml`: set the released model ID or local model
directory, dataset image roots, COCO annotation paths, and output directory.
Relative paths are resolved against the YAML's directory. All sources require
unique `name`, `task` (`detection` or `segmentation`), `images`, and `annotations`.

Detection COCO instances need `bbox` in xywh pixels. Segmentation instances need
polygon or compressed/uncompressed COCO RLE masks; boxes can be omitted and
will be derived from masks. Category IDs can start at 0 and need not be contiguous.
Missing/empty instance masks, invalid boxes, or image-size mismatches fail loudly.
Queries for categories with crowd annotations are excluded, rather than treating
unresolved crowd regions as individual objects or background.

## What is trained

Each sample is an image/query and a complete assistant response:

```text
detection:
  <ref>door</ref><box><100><200><300><500>,<600><200><800><500></box>
segmentation:
  <ref>door</ref><box><100><200><300><500><OBJ>,<600><200><800><500><OBJ></box>
absent target:
  <ref>door</ref><box>None</box>
```

Coordinates are normalized to 0..1000. Prompt, image and system tokens are masked
out of next-token cross entropy. Boxes, OBJ states and mask targets are sorted
together, never matched by rewritten query strings. Binary focal + Dice loss
supervises segmentation only. Detection-only and empty-target samples do not
invoke the pixel decoder. Missing masks are never interpreted as negative masks.

`negative_queries_per_image` adds queries for categories absent from that image.
Only enable this if category annotations are exhaustive. Default source behavior
is 0; the example YAML enables 1. Category sampling within each source is uniform
over image/category query records; source `weight` sets the probability of drawing
from that source. Equal detection/segmentation source weights give a 1:1 mix in
expectation, rather than a strict alternation. Segmentation samples supervise both
tokens and pixels in the same forward; this is downstream fine-tuning, not an
exact reproduction of the paper's alternating large-scale curriculum.

By default, only language-model LoRA and the FP32 pixel decoder are trained.
`train_vision: true` unfreezes the entire visual module (including its mergers).

To train LoRA in both language and vision, use:

```yaml
backbone:
  lora: true
  vision_lora: true
  lora_rank: 16
  lora_alpha: 32
  lora_dropout: 0.05
  train_vision: false
  train_embeddings: false
  attention_implementation: sdpa
```

`vision_lora` defaults to false. When true, adapters are added to every linear
layer inside the Qwen vision transformer blocks (attention QKV/output and MLP
projections). Vision base weights, patch embedding, norms and spatial/deepstack
mergers stay frozen. Language LoRA remains enabled. Both sets of adapters share
`lora_rank`, `lora_alpha` and `lora_dropout`. Vision adapters use
`optimization.vision_learning_rate`; language adapters use
`optimization.learning_rate`. The mask decoder remains fully trained.

`vision_lora: true` requires `lora: true` and `train_vision: false`; conflicting
settings fail before loading the model. Export merges both language and vision
adapters into ordinary weights using the existing export command.

Start a new run/output directory when enabling vision LoRA on a run that did not
have it. This changes model parameters and optimizer groups, so existing training
state cannot be resumed across that change. Existing configurations that omit
`vision_lora` keep their previous behavior and resume signature.

`lora: false` fully tunes the LM; this needs substantially more VRAM.
`train_embeddings: true` with LoRA also trains the full input/output embedding
matrices; it can be expensive. Released special tokens already exist, so these
matrices may remain frozen. Non-reentrant language-model and mask-decoder
checkpointing can be enabled through the `performance` section described in
[OPTIMIZATIONS.md](OPTIMIZATIONS.md). Vision checkpointing remains disabled.

The pixel decoder follows the released FP32 implementation, including its alpha
refinement branch. Here it receives binary-mask supervision. Matting-specific
gradient/Laplacian losses, RL, visual-reference-box training, and cross-image
reference matching augmentation are outside this extension.

## Optional noun-phrase / room-label queries

Add `queries_jsonl` to a source. Each line explicitly maps a query to the complete
set of matching annotation IDs in that source's COCO file:

```json
{"image_id": 123, "query": "door beside room 寝室 (2)", "annotation_ids": [501, 502]}
{"image_id": 123, "query": "window beside room 寝室 (2)", "annotation_ids": []}
```

These records are added to category queries. Japanese labels are preserved.
You must supply exact target IDs; arbitrary `noun_phrases` fields are not guessed
or automatically interpreted. IDs must belong to the specified image and must
not be crowd annotations. An empty ID list is an explicit verified negative.
Reserved model tokens such as `<OBJ>` cannot appear inside a query.

## Check data before loading weights

```bash
python -m wevisone_vl.training --config configs/mixed_task_finetune.yaml --dry-run
```

This checks dataset construction and decodes up to two examples per source,
writing box previews plus exact prompts, responses and annotation IDs into
`OUTPUT_DIR/data_preview/`. It does not test tokenizer lengths, load model weights,
or scan every mask. Training checks every sampled record. Review the source counts
and query targets before launching a long run. Train/validation image-path overlap
is rejected unless explicitly enabled (for intentional overfit debugging).

## Train

```bash
# One GPU (one image/query per microbatch)
CUDA_VISIBLE_DEVICES=0 python -m wevisone_vl.training \
  --config configs/mixed_task_finetune.yaml

# Multiple GPUs using DDP
accelerate launch --multi_gpu --num_processes 2 -m wevisone_vl.training \
  --config configs/mixed_task_finetune.yaml
```

Effective batch size = accumulation steps × process count. Each rank draws the
same source/task but independently samples an image/query. There is no artificial
epoch: training runs for `max_steps` optimizer-update attempts. FP16 overflow can
skip an update; BF16 is the default on supported GPUs. DeepSpeed, quantized training, video inputs and sequence packing are not supported.
FSDP1 is supported for full-parameter detection-only training; see the Rex SFT guide.

For a small overfit check, use one small source, `max_steps: 20`,
`warmup_steps: 0`, and small image/token budgets. With limited VRAM, first lower
`max_pixels`, `max_masks_per_sample`, and LoRA rank, or use the 2B checkpoint.
Gradient accumulation does not reduce one sample's activation memory. Large
drawings should be tiled offline with correspondingly transformed annotations;
this trainer does not slice images. Large target sequences fail rather than
silently dropping instances. `max_masks_per_sample` randomly selects mask
supervision while preserving every target box/OBJ in the text response.
Binary targets use foreground-preserving area resizing: even a thin stroke
marks its output pixel as foreground instead of disappearing under downsampling.

## Logs and validation

Console output and `OUTPUT_DIR/metrics.jsonl` contain training losses and
**generation-based validation quality**. Validation runs greedy generation using
only the image and query. Ground-truth response tokens and boxes are never used
for prediction. Segmentation replays the model's own generated response to obtain
OBJ embeddings, then decodes all predicted masks in chunks with the live weights.
`max_masks_per_sample` limits training supervision only, not generated evaluation.

Add these optional **top-level** YAML keys (defaults shown):

```yaml
checkpoint_metric: auto  # auto | miou | mdice | map | mar
validation_samples_per_source: 32  # 0 = every query; recommended for final selection

generation_validation:
  max_new_tokens: 2048
  mask_threshold: 0.5
  decoder_batch_size: 8
  max_detections: 100
  repetition_penalty: 1.05
```

- Segmentation: `miou` / `mdice` are means of per-query binary foreground-union
  IoU/Dice at original image resolution. Missed and extra foreground is penalized;
  instance order is irrelevant. Empty prediction + empty target scores 1. These
  are semantic coverage metrics, not instance mask AP or instance-matched Dice.
- Detection: `map` is COCO bbox AP averaged over IoU 0.50:0.05:0.95, `mar` is
  COCO AR at `max_detections` over the same thresholds. `map50` / `map75` are
  also logged. AP ranks objects by the geometric mean of the four coordinate
  token probabilities after generation processors. This is an autoregressive
  likelihood proxy, not calibrated detector objectness or the upstream F1 metric.
- Each image/query record is evaluated separately, including explicit phrase
  records. Detection AP averages query categories; overall categories are
  namespaced by source. Segmentation averages queries. Explicit negative queries
  penalize false positives. With the default `negative_queries_per_image: 0`,
  absent category queries are not in the dataset; enable negatives only when
  annotations are exhaustive. Detection subsets without GT have undefined AP/AR
  (`null`), not a fabricated zero.
- Sampling is a fixed seeded random subset per source, identical across validations
  and GPU counts. Distributed workers gather per-query metric records globally;
  AP is computed globally, never averaged across worker APs.

`checkpoint_metric: auto` maximizes `(miou + mdice)/2` for segmentation only,
`(map + mar)/2` for detection only, and the average of these two task scores for
mixed validation. Set `checkpoint_metric: mdice` (for example) to select by one
metric instead. Ties keep the earlier checkpoint. A missing requested metric
raises an error rather than falling back to cross entropy.

`best_checkpoint.txt` points to the best selected score. Independent bests are
also saved as `best_miou_checkpoint.txt`, `best_mdice_checkpoint.txt`,
`best_map_checkpoint.txt`, and `best_mar_checkpoint.txt` for available tasks.
`best_score_checkpoint.txt` mirrors the primary pointer; `best_metrics.json`
records values and paths. Improving any tracked metric triggers a checkpoint,
including between regular save intervals. All tracked bests survive pruning.

Validation events contain `sources`, `overall`, `best_metrics` and `improved`.
`truncated_queries` flags responses reaching the generation token limit (including
an EOS at exactly that limit); increase `max_new_tokens` when necessary.
`malformed_mask_queries` flags OBJ/valid-box alignment failures: those queries
receive an empty prediction rather than incorrectly paired masks. Validation is
slower than a teacher-forced loss pass. Set `validation: []` to disable it.

## Resume and checkpoint storage

```bash
python -m wevisone_vl.training --config configs/mixed_task_finetune.yaml \
  --resume outputs/mixed_task_finetune/checkpoint-0000250
```

Use `latest_checkpoint.txt` or `best_checkpoint.txt` to locate a checkpoint.
Checkpoints include the whole backbone and pixel decoder, optimizer, scheduler,
RNG state, config and training progress. Even LoRA runs save the complete base
weights for reliable resume, so allow several times the model size in disk/RAM.
Keep the original base checkpoint available for reconstruction/export.

Resume verifies config, annotation/query file contents and process count. Keep
`max_steps`, seed, optimizer settings and datasets unchanged for exact resume;
you may change logging/checkpoint intervals or output directory. To extend a
completed schedule, export the model and start a new run with fresh optimizer
state. Image contents are not hashed; do not modify images underneath a run.
The most recent `keep_last` checkpoints plus all independent metric bests are retained.
Existing loss-selected checkpoints can be resumed with the original config and
unchanged training/data settings. The old loss best is discarded for selection;
quality bests start with the first new validation. Updated checkpoints persist
all quality bests and include the new validation settings in the resume signature.
Do not change those settings partway through an updated run. Old `best_loss` is
retained in state for compatibility only and is not used for selection.

When resuming into a different output directory, the best pointer may refer to
the original run until validation improves; keep that original checkpoint.

## Export and use with the original inference API

```bash
python -m wevisone_vl.training.export \
  --checkpoint outputs/mixed_task_finetune/checkpoint-0002000 \
  --output outputs/mixed_task_export
```

Export loads on CPU and merges LoRA. It writes a complete Transformers model,
processor/tokenizer and compatible `mask_decoder.pt`. Allow RAM for both loaded
weights and the checkpoint state during export. The output directory must be new
or empty. Training-state directories are not directly inference model directories.

```python
from wevisone_vl import WeVisOneVL

model = WeVisOneVL("outputs/mixed_task_export")
result = model.predict("drawing.png", task="detection", query=["door", "window"])
result = model.predict("drawing.png", task="segmentation", query="specialty")
# The released engine returns probabilities; threshold masks for binary output.
binary_masks = [mask >= 0.5 for mask in result.masks]
```

## Verification

```bash
OMP_NUM_THREADS=2 python -m pytest -q
```

Tests cover COCO targets, polygon/RLE masks, assistant-only labels, gradient flow,
the real pixel decoder, and a tiny randomly initialized Qwen3-VL with real LoRA.
Integration checks exercise optimizer steps, validation, interrupted/resumed
training, merged export and the original inference loader, without downloading
model weights. They do not establish full-size GPU memory use or task accuracy.
