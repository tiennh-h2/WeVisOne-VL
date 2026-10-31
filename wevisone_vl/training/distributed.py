"""FSDP1 policy and checkpoint helpers for full-parameter detection SFT."""
from functools import partial
from pathlib import Path


def is_fsdp(accelerator):
    return accelerator.distributed_type.name == 'FSDP'


def configure_fsdp(accelerator, model, cfg):
    if not is_fsdp(accelerator):
        return
    from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig, ShardedOptimStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    plugin = accelerator.state.fsdp_plugin
    if cfg['backbone']['lora']:
        raise ValueError('FSDP currently requires full backbone training (lora: false)')
    if any(s['task'] != 'detection' for s in cfg['train'] + cfg['validation']):
        raise ValueError('FSDP currently supports detection-only training and validation')
    if getattr(plugin, 'fsdp_version', 1) != 1:
        raise ValueError('this trainer requires FSDP1; set fsdp_version: 1 in the launch config')
    if not plugin.use_orig_params:
        raise ValueError('FSDP requires fsdp_use_orig_params: true for optimizer parameter groups')
    if plugin.cpu_ram_efficient_loading:
        raise ValueError('set fsdp_cpu_ram_efficient_loading: false; the custom decoder must load on every rank')
    if getattr(plugin, 'activation_checkpointing', False):
        raise ValueError('use performance language/vision_checkpointing, not FSDP activation_checkpointing')
    root = model.backbone.model
    classes = {type(block) for block in root.visual.blocks}
    classes.update(type(block) for block in root.language_model.layers)
    plugin.auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=classes)
    plugin.state_dict_type = StateDictType.SHARDED_STATE_DICT
    plugin.state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
    plugin.optim_state_dict_config = ShardedOptimStateDictConfig(offload_to_cpu=True)
    # Clip through accelerator.clip_grad_norm_: FSDP computes a global norm.
    # Sync every microbatch: FSDP no_sync would accumulate FULL gradients.
    accelerator.gradient_state.plugin_kwargs['sync_each_batch'] = True


def generation_forward(model, inputs):
    output = model({'generation_step': True, 'inputs': inputs})
    return output.logits[:, -1, :], output.past_key_values


def resolve_checkpoint_weights(checkpoint, merge=None):
    checkpoint = Path(checkpoint)
    full = checkpoint / 'pytorch_model.bin'
    if full.is_file():
        return full
    shard = checkpoint / 'pytorch_model_fsdp_0'
    if not shard.is_dir():
        raise FileNotFoundError(f'no full or sharded model weights found in {checkpoint}')
    destination = checkpoint / 'merged_model'
    full = destination / 'pytorch_model.bin'
    if not full.is_file():
        if merge is None:
            from accelerate.utils import merge_fsdp_weights
            merge = merge_fsdp_weights
        destination.mkdir(parents=True, exist_ok=True)
        merge(str(shard), str(destination), safe_serialization=False)
    return full
