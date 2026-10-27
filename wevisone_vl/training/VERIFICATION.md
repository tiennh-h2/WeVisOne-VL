# Verification record — generated quality validation

Tested on CPU with Python 3.12 and PyTorch 2.14.1+cpu. No pretrained weights
were downloaded. The uploaded archive supplied the training folder only; it did
not contain the earlier training tests mentioned in the original README.

The updated folder was overlaid into a fresh upstream WeVisOne-VL source tree.
Full-suite command, from that tree:

```bash
OMP_NUM_THREADS=2 PYTHONPATH=. python -m pytest -q
```

Result: 36 passed, 2 pycocotools/NumPy deprecation warnings. This includes 20
upstream CLI/evaluation tests and 16 tests in `wevisone_vl/training/tests/`.
Python compilation also passed.

Added coverage:
- Real tiny Qwen3-VL + PEFT vision LoRA: trainable visual attention/MLP adapters,
  frozen base weights and mergers, nonzero adapter gradients, vision learning-rate
  grouping, and merge equivalence after an optimizer step.
- Existing frozen/full-vision modes and invalid vision LoRA config combinations.
- Perfect/missed/extra segmentation foreground and empty targets.
- COCO AP/AR for perfect/missing boxes, duplicate predictions, false positives on
  negative queries, configurable max detections and undefined all-negative AP.
- Generated coordinate parsing, token probability ranking and absent responses.
- Task-balanced selection score, independent metric bests and tie behavior.
- Real CPU Accelerate optimizer/checkpoint save/load, resume and best retention.
- Legacy loss-selected state migration and config/data signature guards.
- Tiny synthetic VLM/decoder with real feature capture hooks: generation sees only
  the query prefix; mask decoding receives generated coordinates; original model
  training mode is restored. No GT tokens enter the generated prediction path.

Not verified: released 2B/4B model inference, real dataset quality, GPU memory or
latency, CUDA mixed precision, multi-process DDP, or merged export after this
change. The synthetic generation test checks plumbing rather than model accuracy.
The earlier archive's verification claims are superseded by this record.
