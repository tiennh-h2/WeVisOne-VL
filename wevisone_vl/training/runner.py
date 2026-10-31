"""Accelerate training loop with deterministic task sampling and resumable state."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import time
from threading import Lock
from pathlib import Path

import torch
import yaml

from .collator import TrainingCollator
from .data import CocoTaskDataset
from .model import MixedTaskModel, configure_backbone, configure_language_checkpointing, configure_vision_checkpointing
from .performance import DEFAULTS as PERFORMANCE_DEFAULTS, prefetch
from .quality import update_bests
from .recipe import apply_recipe
from .sampling import EpochSchedule
from .distributed import is_fsdp, configure_fsdp
from .validation import evaluate_generation


def validate_config(cfg: dict) -> None:
    apply_recipe(cfg)
    defaults = {
        "seed": 42, "max_steps": 2000, "validation": [], "revision": None,
        "max_length": 8192, "min_pixels": 16 * 32 * 32, "max_pixels": 1024 * 1024,
        "max_masks_per_sample": 24, "save_every": 250, "validate_every": 250,
        "validation_samples_per_source": 32, "log_every": 10, "keep_last": 2,
        "allow_validation_overlap": False, "checkpoint_metric": "auto",
        "generation_validation": {}, "performance": {},
        "recipe": "legacy", "sampling": "weighted", "num_train_epochs": 1,
        "loss_normalization": "queries",
    }
    unknown = set(cfg) - set(defaults) - {"model", "output_dir", "train", "optimization", "backbone", "loss"}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    for name, value in defaults.items():
        cfg.setdefault(name, value)
    if cfg["checkpoint_metric"] not in {"auto", "miou", "mdice", "map", "mar"}:
        raise ValueError("checkpoint_metric must be auto, miou, mdice, map or mar")
    sections = {
        "generation_validation": {"max_new_tokens": 2048, "mask_threshold": .5,
                                  "decoder_batch_size": 8, "max_detections": 100,
                                  "repetition_penalty": 1.05},
        "optimization": {"gradient_accumulation_steps": 8, "mixed_precision": "bf16",
                         "learning_rate": 5e-6, "decoder_learning_rate": 1e-5,
                         "vision_learning_rate": 5e-7, "weight_decay": .01,
                         "merger_learning_rate": None, "warmup_ratio": None,
                         "max_grad_norm": 1., "warmup_steps": min(100, cfg["max_steps"] // 20)},
        "backbone": {"lora": True, "lora_rank": 16, "lora_alpha": 32,
                     "lora_dropout": .05, "train_vision": False,
                     "train_embeddings": False, "attention_implementation": "sdpa"},
        "loss": {"ntp_weight": 1., "segmentation_weight": 10., "focal_weight": 20., "dice_weight": 1.},
    }
    for name, values in sections.items():
        section = cfg.setdefault(name, {})
        allowed = set(values) | ({'vision_lora', 'train_merger'} if name == 'backbone' else set())
        if set(section) - allowed:
            raise ValueError(f"unknown {name} keys: {sorted(set(section) - allowed)}")
        for key, value in values.items():
            section.setdefault(key, value)
    performance = cfg["performance"]
    if not isinstance(performance, dict) or set(performance) - set(PERFORMANCE_DEFAULTS):
        raise ValueError("invalid performance settings")
    for key, default in PERFORMANCE_DEFAULTS.items():
        performance.setdefault(key, default)
        value = performance[key]
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"performance.{key} must be boolean")
    if performance["fused_adamw"] != "auto" and type(performance["fused_adamw"]) is not bool:
        raise ValueError("performance.fused_adamw must be auto, true or false")
    depth = performance["prefetch_batches"]
    if type(depth) is not int or depth < 0:
        raise ValueError("performance.prefetch_batches must be an integer >= 0")
    chunk_size = performance["refinement_chunk_size"]
    if type(chunk_size) is not int or chunk_size < 0:
        raise ValueError("performance.refinement_chunk_size must be an integer >= 0")
    backbone = cfg['backbone']
    vision_lora = backbone.get('vision_lora', False)
    if not isinstance(vision_lora, bool):
        raise ValueError('vision_lora must be a boolean')
    if vision_lora and (not backbone['lora'] or backbone['train_vision']):
        raise ValueError('vision_lora requires lora: true and train_vision: false')
    if cfg['sampling'] not in {'weighted', 'epochs'}:
        raise ValueError('sampling must be weighted or epochs')
    if type(cfg['num_train_epochs']) is not int or cfg['num_train_epochs'] <= 0:
        raise ValueError('num_train_epochs must be a positive integer')
    if cfg['loss_normalization'] not in {'tokens', 'queries'}:
        raise ValueError('loss_normalization must be tokens or queries')
    if cfg['sampling'] == 'epochs' and any(float(s.get('weight', 1)) != 1 for s in cfg['train']):
        raise ValueError('source weights are only used by weighted sampling; epochs require weight: 1')
    if cfg.get('recipe') == 'rex_sft' and any(s['task'] != 'detection' for s in cfg['train']):
        raise ValueError('rex_sft recipe requires detection training sources')
    if 'train_merger' in backbone and type(backbone['train_merger']) is not bool:
        raise ValueError('train_merger must be boolean')
    for key in ("model", "output_dir", "train"):
        if not cfg.get(key):
            raise ValueError(f"config requires {key}")
    for key in ("max_steps", "max_length", "min_pixels", "max_pixels", "save_every",
                "validate_every", "log_every", "keep_last"):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    limit = cfg['validation_samples_per_source']
    if not isinstance(limit, int) or limit < 0:
        raise ValueError('validation_samples_per_source must be >= 0 (0 means all)')
    gen = cfg['generation_validation']
    for key in ('max_new_tokens', 'decoder_batch_size', 'max_detections'):
        if not isinstance(gen[key], int) or gen[key] <= 0:
            raise ValueError(f'generation_validation.{key} must be a positive integer')
    for key in ('mask_threshold', 'repetition_penalty'):
        gen[key] = float(gen[key])
        if not math.isfinite(gen[key]):
            raise ValueError(f'generation_validation.{key} must be finite')
    if not 0 < gen['mask_threshold'] < 1 or gen['repetition_penalty'] <= 0:
        raise ValueError('mask_threshold must be in (0,1); repetition_penalty must be positive')
    if cfg["min_pixels"] > cfg["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    opt = cfg["optimization"]
    for key in ("learning_rate", "decoder_learning_rate", "vision_learning_rate", "weight_decay", "max_grad_norm"):
        opt[key] = float(opt[key])
    if opt['merger_learning_rate'] is not None:
        opt['merger_learning_rate'] = float(opt['merger_learning_rate'])
        if not math.isfinite(opt['merger_learning_rate']) or opt['merger_learning_rate'] <= 0:
            raise ValueError('merger_learning_rate must be finite and positive')
    if opt['warmup_ratio'] is not None and not 0 <= float(opt['warmup_ratio']) < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    for key in cfg["loss"]:
        cfg["loss"][key] = float(cfg["loss"][key])
    if not isinstance(opt["gradient_accumulation_steps"], int) or opt["gradient_accumulation_steps"] <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    if opt["mixed_precision"] not in {"no", "bf16", "fp16"}:
        raise ValueError("mixed_precision must be no, bf16 or fp16")
    for key in ("learning_rate", "decoder_learning_rate", "vision_learning_rate", "max_grad_norm"):
        if not math.isfinite(float(opt[key])) or float(opt[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not 0 <= opt["warmup_steps"] < cfg["max_steps"]:
        raise ValueError("warmup_steps must be >= 0 and less than max_steps")
    if any(not math.isfinite(float(v)) or v < 0 for v in cfg["loss"].values()):
        raise ValueError("loss weights must be finite and nonnegative")
    if cfg["loss"]["ntp_weight"] <= 0:
        raise ValueError("ntp_weight must be positive for mixed training")
    for split in ("train", "validation"):
        names = set()
        if not isinstance(cfg[split], list):
            raise ValueError(f"{split} must be a list of dataset sources")
        for source in cfg[split]:
            unknown = set(source) - {"name", "task", "images", "annotations", "weight",
                                     "category_ids", "exclude_category_ids",
                                     "negative_queries_per_image", "queries_jsonl", "query_mode", "phrase_sampling"}
            if unknown:
                raise ValueError(f"unknown source keys: {sorted(unknown)}")
            if any(not source.get(k) for k in ("name", "task", "images", "annotations")):
                raise ValueError("each source needs name, task, images and annotations")
            if source["name"] in names:
                raise ValueError(f"duplicate source name in {split}: {source['name']}")
            names.add(source["name"])
            if source["task"] not in {"detection", "segmentation"}:
                raise ValueError("source task must be detection or segmentation")
            if source.get('query_mode', 'single') not in {'single', 'multi'}:
                raise ValueError('query_mode must be single or multi')
            if source.get('phrase_sampling', 'subset') not in {'subset', 'all'}:
                raise ValueError('phrase_sampling must be subset or all')
            if source.get('query_mode') == 'multi' and (split != 'train' or source['task'] != 'detection'):
                raise ValueError('multi queries are supported for detection training only; validation uses single queries')
            weight = float(source.get("weight", 1.))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("source weight must be finite and positive")


def load_config(path) -> dict:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    validate_config(cfg)
    for split in ("train", "validation"):
        for source in cfg[split]:
            for key in ("images", "annotations", "queries_jsonl"):
                if source.get(key):
                    source[key] = str((path.parent / Path(source[key]).expanduser()).resolve())
    cfg["output_dir"] = str((path.parent / Path(cfg["output_dir"]).expanduser()).resolve())
    model_path = Path(cfg["model"]).expanduser()
    if model_path.is_absolute() or cfg["model"].startswith(".") or (path.parent / model_path).exists():
        cfg["model"] = str((path.parent / model_path).resolve())
    return cfg


def resume_signature(cfg, legacy=False) -> str:
    value = {k: v for k, v in cfg.items() if k not in
             {"output_dir", "save_every", "validate_every", "log_every", "keep_last", "performance"}
             and not (legacy and k in {"checkpoint_metric", "generation_validation"})}
    if cfg.get('recipe', 'legacy') == 'legacy':
        # Keep signatures of pre-FSDP runs when all added options are inactive.
        value = copy.deepcopy(value)
        for key, default in [('recipe', 'legacy'), ('sampling', 'weighted'),
                             ('num_train_epochs', 1), ('loss_normalization', 'queries')]:
            if value.get(key) == default:
                value.pop(key, None)
        for key in ['merger_learning_rate', 'warmup_ratio']:
            if value.get('optimization', {}).get(key) is None:
                value.get('optimization', {}).pop(key, None)
    digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode())
    for source in cfg["train"] + cfg.get("validation", []):
        for key in ("annotations", "queries_jsonl"):
            if source.get(key):
                with Path(source[key]).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
    return digest.hexdigest()


def choose_record(datasets, weights, seed: int, micro_step: int, rank: int) -> tuple[int, int]:
    # Every rank uses the same task/source, with independent record selection.
    source = random.Random(seed + micro_step * 1000003).choices(range(len(datasets)), weights=weights)[0]
    index = random.Random(seed + micro_step * 1000033 + rank * 1000000007).randrange(len(datasets[source]))
    return source, index


def load_components(cfg):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from ..config import Qwen3VLSegConfig
    from ..decoder import build_mask_decoder_from_qwen
    from ..engine import _resolve_sidecar

    sidecar = torch.load(_resolve_sidecar(cfg["model"], cfg["revision"]), map_location="cpu", weights_only=True)
    decoder_cfg = Qwen3VLSegConfig.from_dict(sidecar.get("cfg"))
    processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["revision"],
                                             min_pixels=cfg["min_pixels"], max_pixels=cfg["max_pixels"],
                                             trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    vocabulary = processor.tokenizer.get_vocab()
    required = ["<OBJ>"] + [f"<{n}>" for n in range(1001)]
    if any(token not in vocabulary for token in required):
        raise ValueError("checkpoint must contain released OBJ and coordinate tokens")
    seg_id = processor.tokenizer.convert_tokens_to_ids("<OBJ>")
    if sidecar.get("seg_token_id", seg_id) != seg_id:
        raise ValueError("sidecar/tokenizer OBJ token mismatch")
    # Trainable parameters stay FP32 unless frozen BF16 base weights can be used.
    base_dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    dtype = base_dtypes[cfg["optimization"]["mixed_precision"]] if cfg["backbone"]["lora"] else torch.float32
    vlm = AutoModelForImageTextToText.from_pretrained(
        cfg["model"], revision=cfg["revision"], dtype=dtype,
        attn_implementation=cfg["backbone"]["attention_implementation"], trust_remote_code=True)
    vlm.config.use_cache = False
    # Temporary feature hooks cannot safely be recomputed by reentrant checkpointing.
    if hasattr(vlm, "gradient_checkpointing_disable"):
        vlm.gradient_checkpointing_disable()
    merge_size = int(sidecar.get("spatial_merge_size") or vlm.config.vision_config.spatial_merge_size)
    decoder = build_mask_decoder_from_qwen(vlm, decoder_cfg)
    decoder.load_state_dict(sidecar["state_dict"], strict=True)
    vlm = configure_backbone(vlm, cfg["backbone"])
    performance = cfg.get("performance", PERFORMANCE_DEFAULTS)
    configure_language_checkpointing(vlm, performance["language_checkpointing"])
    if performance['vision_checkpointing'] and any(s['task'] == 'segmentation' for s in cfg['train']):
        raise ValueError('vision_checkpointing is currently detection-only; segmentation feature hooks need original execution')
    configure_vision_checkpointing(vlm, performance['vision_checkpointing'])
    if cfg.get('recipe') == 'rex_sft' and not any(s['task'] == 'segmentation' for s in cfg['train']):
        decoder.requires_grad_(False)
    for parameter in vlm.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = MixedTaskModel(vlm, decoder, decoder_cfg, seg_id, merge_size,
                           max_masks_per_sample=cfg["max_masks_per_sample"],
                           selective_logits=performance["selective_logits"],
                           decoder_checkpointing=performance["decoder_checkpointing"],
                           refinement_chunk_size=performance["refinement_chunk_size"], **cfg["loss"])
    collator = TrainingCollator(processor, seg_id, decoder_cfg.cnn_image_size, cfg["max_length"],
                                fast_prefix=performance["fast_prefix"])
    return model, processor, collator


def build_datasets(cfg):
    train = [CocoTaskDataset(source, cfg["seed"]) for source in cfg["train"]]
    valid = [CocoTaskDataset(source, cfg["seed"]) for source in cfg["validation"]]
    if valid and not cfg["allow_validation_overlap"]:
        def paths(datasets):
            return {(ds.root / ds.images[r["image_id"]]["file_name"]).resolve()
                    for ds in datasets for r in ds.records}
        overlap = paths(train) & paths(valid)
        if overlap:
            raise ValueError(f"train/validation image overlap: {next(iter(overlap))}")
    return train, valid


METRIC_KEYS = ("loss", "ntp_loss", "seg_loss", "focal_loss", "dice_loss", "mask_dice")


def metric_vector(output, device):
    values = torch.stack([output[key].detach().to(device=device, dtype=torch.float64)
                          for key in METRIC_KEYS])
    values[-1] *= output["mask_count"]
    return torch.cat((values, values.new_tensor([output["mask_count"], 1.])))


def metric_dict(vector):
    values = vector.tolist()
    count = max(values[-1], 1.)
    result = {key: values[i] / (max(values[-2], 1.) if key == "mask_dice" else count)
              for i, key in enumerate(METRIC_KEYS)}
    result.update(mask_count=int(values[-2]), query_count=int(values[-1]))
    return result


@torch.no_grad()
def evaluate(model, datasets, collator, accelerator, limit):
    from accelerate.utils import send_to_device

    model.eval()
    rows = {}
    total = torch.zeros(8, dtype=torch.float64, device=accelerator.device)
    # Bypass DDP for variable-length validation shards: no backward/collectives in forward.
    unwrapped = accelerator.unwrap_model(model)
    for dataset in datasets:
        vector = torch.zeros_like(total)
        for index in range(accelerator.process_index, (min(len(dataset), limit) if limit else len(dataset)), accelerator.num_processes):
            batch = send_to_device(collator([dataset[index]]), accelerator.device)
            with accelerator.autocast():
                output = unwrapped(batch)
            vector += metric_vector(output, accelerator.device)
        vector = accelerator.reduce(vector, reduction="sum")
        rows[dataset.name] = metric_dict(vector)
        total += vector
    rows["overall"] = metric_dict(total)
    model.train()
    return rows


def make_optimizer(model, cfg):
    opt = cfg["optimization"]
    groups = {}
    visual = model.backbone.model.visual
    vision_ids = {id(p) for p in visual.parameters()}
    merger_ids = {id(p) for name, module in visual.named_modules()
                  if 'merger' in name.split('.')[-1]
                  for p in module.parameters()}
    if opt.get('merger_learning_rate') is None and 'train_merger' not in cfg['backbone']:
        merger_ids = set()
    rates = {'language': opt['learning_rate'], 'vision': opt['vision_learning_rate'],
             'merger': opt.get('merger_learning_rate') or opt['vision_learning_rate'],
             'decoder': opt['decoder_learning_rate']}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = ('decoder' if name.startswith('mask_decoder.') else
                 'merger' if id(parameter) in merger_ids else
                 'vision' if id(parameter) in vision_ids else 'language')
        # Rex uses separate decay/no-decay groups. Preserve legacy behavior.
        decay = (parameter.ndim > 1 and not name.endswith('.bias')) if cfg.get('recipe') == 'rex_sft' else True
        groups.setdefault((group, decay), []).append(parameter)
    parameter_groups = [{'params': params, 'lr': rates[key], 'name': key,
                         'weight_decay': opt['weight_decay'] if decay else 0.}
                        for (key, decay), params in sorted(groups.items(), key=lambda item:
                            (['language','vision','merger','decoder'].index(item[0][0]), not item[0][1]))]
    mode = cfg.get("performance", PERFORMANCE_DEFAULTS)["fused_adamw"]
    cuda = all(p.device.type == "cuda" for group in parameter_groups for p in group["params"])
    import inspect
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if mode is True and (not cuda or not supports_fused):
        raise ValueError("fused AdamW requires CUDA parameters and a supporting PyTorch version")
    kwargs = {"fused": mode is not False and cuda} if supports_fused else {}
    return torch.optim.AdamW(parameter_groups, weight_decay=opt["weight_decay"], **kwargs)


def train(cfg, resume=None):
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import send_to_device, set_seed

    opt = cfg["optimization"]
    accelerator = Accelerator(mixed_precision=opt["mixed_precision"],
                              gradient_accumulation_steps=opt["gradient_accumulation_steps"],
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    if accelerator.distributed_type.name not in {'NO', 'MULTI_GPU', 'MULTI_CPU', 'FSDP'}:
        raise ValueError('supported backends: single-process, DDP, FSDP1')
    if is_fsdp(accelerator):
        # Check before from_pretrained: RAM-efficient loading can leave nonzero
        # ranks with empty/meta weights, which our custom sidecar does not support.
        plugin = accelerator.state.fsdp_plugin
        if getattr(plugin, 'fsdp_version', 1) != 1 or plugin.cpu_ram_efficient_loading:
            raise ValueError('use fsdp_version: 1 and fsdp_cpu_ram_efficient_loading: false')
        if cfg['backbone']['lora'] or any(ds['task'] != 'detection' for ds in cfg['train'] + cfg['validation']):
            raise ValueError('FSDP requires full backbone training and detection-only sources')
    if opt["mixed_precision"] == "bf16" and accelerator.device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported by this GPU; choose mixed_precision: no or fp16")
    set_seed(cfg["seed"], device_specific=True)
    datasets, validation = build_datasets(cfg)
    schedule = None
    if cfg['sampling'] == 'epochs':
        schedule = EpochSchedule([len(ds) for ds in datasets], cfg['num_train_epochs'],
                                 accelerator.num_processes, opt['gradient_accumulation_steps'], cfg['seed'])
        cfg['max_steps'] = schedule.max_steps
    if opt['warmup_ratio'] is not None:
        opt['warmup_steps'] = math.ceil(float(opt['warmup_ratio']) * cfg['max_steps'])
        opt['warmup_steps'] = min(opt['warmup_steps'], cfg['max_steps'] - 1)
    signature = resume_signature(cfg)
    output_dir = Path(cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not resume and (output_dir / "latest_checkpoint.txt").exists():
            raise ValueError("output_dir already contains training; use --resume or a new output_dir")
    accelerator.wait_for_everyone()
    model, processor, collator = load_components(cfg)
    configure_fsdp(accelerator, model, cfg)
    if not is_fsdp(accelerator):
        model.to(accelerator.device)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, cfg)
    warmup = opt["warmup_steps"]
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = min(1., (step - warmup) / max(cfg["max_steps"] - warmup, 1))
        return .5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    model, optimizer = accelerator.prepare(model, optimizer)
    accelerator.register_for_checkpointing(scheduler)
    start_step, best_loss, best_checkpoint = 0, None, None
    best_metrics = {}
    if resume:
        resume = Path(resume).expanduser().resolve()
        state = json.loads((resume / "training_state.json").read_text())
        valid_signatures = {signature}
        if 'best_metrics' not in state:
            valid_signatures.add(resume_signature(cfg, legacy=True))
        if state["signature"] not in valid_signatures or state["world_size"] != accelerator.num_processes:
            raise ValueError("resume config/data/world size changed; use the original config or start a new run")
        backend = state.get('backend', 'ddp')
        if (backend == 'fsdp') != is_fsdp(accelerator):
            raise ValueError('resume backend changed; export weights and start a new run instead')
        accelerator.load_state(str(resume))
        start_step, best_loss, best_checkpoint = state["step"], state["best_loss"], state["best_checkpoint"]
        # Old loss-selected checkpoints are never treated as quality-selected bests.
        best_metrics = state.get('best_metrics', {})
        for entry in best_metrics.values():
            if not Path(entry['checkpoint']).is_absolute():
                entry['checkpoint'] = str((resume.parent / entry['checkpoint']).resolve())
        best_checkpoint = best_metrics.get('score', {}).get('checkpoint')
    if start_step >= cfg["max_steps"]:
        raise ValueError("checkpoint already reached max_steps")

    def log(row):
        if accelerator.is_main_process:
            line = json.dumps(row, ensure_ascii=False, allow_nan=False)
            print(line, flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def save(step):
        directory = output_dir / f"checkpoint-{step:07d}"
        accelerator.save_state(str(directory), safe_serialization=False)
        if accelerator.is_main_process:
            (directory / "training_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            state = {"step": step, "signature": signature, "world_size": accelerator.num_processes,
                     "best_loss": best_loss, "best_checkpoint": best_checkpoint,
                     "best_metrics": best_metrics,
                     "backend": 'fsdp' if is_fsdp(accelerator) else 'ddp',
                     "checkpoint_format": 'sharded' if is_fsdp(accelerator) else 'full'}
            (directory / "training_state.json").write_text(json.dumps(state, indent=2))
            (output_dir / "latest_checkpoint.txt").write_text(str(directory))
            if best_checkpoint:
                (output_dir / "best_checkpoint.txt").write_text(best_checkpoint)
            else:
                (output_dir / "best_checkpoint.txt").unlink(missing_ok=True)
            for key, entry in best_metrics.items():
                (output_dir / f'best_{key}_checkpoint.txt').write_text(entry['checkpoint'])
            (output_dir / 'best_metrics.json').write_text(json.dumps(best_metrics, indent=2))
            protected = {entry['checkpoint'] for entry in best_metrics.values()}
            existing = sorted(output_dir.glob("checkpoint-*"))
            removable = [path for path in existing[:-cfg["keep_last"]]
                         if str(path.resolve()) not in protected and (path / "training_state.json").is_file()]
            for path in removable:
                shutil.rmtree(path)
        accelerator.wait_for_everyone()

    log({"event": "start", "step": start_step, "world_size": accelerator.num_processes,
         "trainable_parameters": trainable_count,
         "backend": accelerator.distributed_type.name, "sampling": cfg["sampling"],
         "max_steps": cfg["max_steps"], "warmup_steps": opt["warmup_steps"],
         "train_queries": {ds.name: len(ds) for ds in datasets},
         "validation_queries": {ds.name: len(ds) for ds in validation},
         "validation": "free-running generation: foreground mIoU/mDice and COCO bbox mAP/mAR",
         "vision_training": ("lora" if cfg['backbone'].get('vision_lora', False)
                             else "full" if cfg['backbone']['train_vision'] else "frozen"),
         "checkpoint_metric": cfg["checkpoint_metric"],
         "effective_batch_size": opt["gradient_accumulation_steps"] * accelerator.num_processes})
    weights = [float(source.get("weight", 1)) for source in cfg["train"]]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window = {task: torch.zeros(8, dtype=torch.float64, device=accelerator.device)
              for task in ("detection", "segmentation")}
    performance = cfg.get("performance", PERFORMANCE_DEFAULTS)
    processor_lock = Lock()
    def prepare_microbatch(micro_step):
        if schedule is not None:
            step, micro = divmod(micro_step, opt['gradient_accumulation_steps'])
            source, index, epoch, valid = schedule.record(step, micro, accelerator.process_index)
        else:
            source, index = choose_record(datasets, weights, cfg['seed'], micro_step, accelerator.process_index)
            epoch, valid = micro_step, True
        sample = datasets[source].sample(index, epoch=epoch)
        with processor_lock:
            batch = collator([sample])
        batch['sample_weight'] = float(valid)
        return batch
    microbatches = prefetch(
        range(start_step * opt["gradient_accumulation_steps"], cfg["max_steps"] * opt["gradient_accumulation_steps"]),
        prepare_microbatch, performance["prefetch_batches"])
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    window_start = time.perf_counter()
    window_steps = 0
    for step in range(start_step, cfg["max_steps"]):
        # Normalize over all real target tokens in this global optimizer batch,
        # including unequal response lengths and the padded last epoch batch.
        batches = [next(microbatches) for _ in range(opt['gradient_accumulation_steps'])]
        counts = [int((batch['inputs']['labels'][:, 1:] != -100).sum()) * batch['sample_weight']
                  for batch in batches]
        totals = torch.tensor([sum(counts), sum(b['sample_weight'] for b in batches)],
                              device=accelerator.device, dtype=torch.float64)
        totals = accelerator.reduce(totals, reduction='sum')
        denominator = totals[0 if cfg['loss_normalization'] == 'tokens' else 1].item()
        for batch, token_count in zip(batches, counts):
            numerator = token_count if cfg['loss_normalization'] == 'tokens' else batch['sample_weight']
            scale = numerator * accelerator.num_processes * opt['gradient_accumulation_steps'] / denominator
            batch = send_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                result = model(batch)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError(f"nonfinite loss in source {batch['source']}, image {batch['image_id']}")
                accelerator.backward(result['loss'] * scale)
                window[batch['task']] += metric_vector(result, accelerator.device) * batch['sample_weight']
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt["max_grad_norm"])
                optimizer.step()
                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        window_steps += 1
        if completed % cfg["log_every"] == 0 or completed == cfg["max_steps"]:
            metrics = {}
            for task in sorted(window):
                vector = accelerator.reduce(window[task], reduction="sum")
                if vector[-1] > 0:
                    metrics[task] = metric_dict(vector)
                window[task].zero_()
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize(accelerator.device)
            elapsed = time.perf_counter() - window_start
            timing = {"seconds_per_step": elapsed / window_steps,
                      "queries_per_second": window_steps * opt["gradient_accumulation_steps"] * accelerator.num_processes / elapsed}
            if accelerator.device.type == "cuda":
                timing.update(peak_allocated_gib=torch.cuda.max_memory_allocated(accelerator.device) / 2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved(accelerator.device) / 2**30)
            log({"event": "train", "step": completed, "tasks": metrics,
                 "learning_rates": scheduler.get_last_lr(), "performance": timing})
            window_start, window_steps = time.perf_counter(), 0
        pause_start = time.perf_counter()
        improved = False
        if validation and (completed % cfg["validate_every"] == 0 or completed == cfg["max_steps"]):
            # Fast tokenizers mutate padding/truncation state during calls.
            # Do not share a processor concurrently with the prefetch worker.
            with processor_lock:
                metrics = evaluate_generation(model, validation, processor, accelerator, cfg)
            checkpoint = str((output_dir / f"checkpoint-{completed:07d}").resolve())
            updated = update_bests(best_metrics, metrics['overall'], checkpoint)
            improved = updated != best_metrics
            best_metrics = updated
            best_checkpoint = best_metrics['score']['checkpoint']
            log({"event": "validation", "step": completed, **metrics,
                 "checkpoint_metric": cfg['checkpoint_metric'],
                 "best_metrics": best_metrics, "improved": improved})
        if improved or completed % cfg["save_every"] == 0 or completed == cfg["max_steps"]:
            save(completed)
        window_start += time.perf_counter() - pause_start
        if validation and (completed % cfg["validate_every"] == 0 or completed == cfg["max_steps"]):
            if accelerator.device.type == "cuda":
                torch.cuda.synchronize(accelerator.device)
                torch.cuda.reset_peak_memory_stats(accelerator.device)
    accelerator.wait_for_everyone()
    accelerator.end_training()


def dry_run(cfg, count=2):
    from PIL import ImageDraw

    datasets, validation = build_datasets(cfg)
    directory = Path(cfg["output_dir"]) / "data_preview"
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, sources in (("train", datasets), ("validation", validation)):
        for source_number, dataset in enumerate(sources):
            for index in range(min(count, len(dataset))):
                sample = dataset[index]
                image = sample["image"].copy()
                draw = ImageDraw.Draw(image)
                for number, box in enumerate(sample["boxes"]):
                    xyxy = [box[0]*image.width, box[1]*image.height,
                            box[2]*image.width, box[3]*image.height]
                    draw.rectangle(xyxy, outline="red", width=3)
                    draw.text((xyxy[0], xyxy[1]), str(number), fill="red")
                image.thumbnail((1024, 1024))
                name = f"{split}-{source_number}-{index}.jpg"
                image.save(directory / name)
                row = {key: sample[key] for key in ("source", "task", "image_id", "file_name",
                       "query", "annotation_ids", "prompt", "response", "boxes")}
                row.update(split=split, preview=name)
                rows.append(row)
    (directory / "samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train_queries": {ds.name: len(ds) for ds in datasets},
                      "validation_queries": {ds.name: len(ds) for ds in validation},
                      "preview": str(directory)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="checkpoint directory, not an inference export")
    parser.add_argument("--dry-run", action="store_true", help="check data and dump queries without loading model")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg)
    else:
        train(cfg, args.resume)
