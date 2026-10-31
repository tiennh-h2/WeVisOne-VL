"""Opt-in Rex SFT defaults; never rewrite original model/token configuration."""


def apply_recipe(cfg):
    recipe = cfg.get('recipe', 'legacy')
    if recipe not in {'legacy', 'rex_sft'}:
        raise ValueError('recipe must be legacy or rex_sft')
    if recipe != 'rex_sft':
        return
    defaults = dict(sampling='epochs', num_train_epochs=1, loss_normalization='tokens')
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    sections = {
        'backbone': dict(lora=False, train_vision=True, train_merger=True, train_embeddings=True),
        'optimization': dict(learning_rate=2e-5, vision_learning_rate=2e-6,
                             merger_learning_rate=2e-5, warmup_ratio=.03),
        'loss': dict(ntp_weight=1., segmentation_weight=0.),
        'performance': dict(language_checkpointing=True, vision_checkpointing=True),
    }
    for section, values in sections.items():
        for key, value in values.items():
            cfg.setdefault(section, {}).setdefault(key, value)
    for source in cfg.get('train', []):
        source.setdefault('query_mode', 'multi')
        source.setdefault('phrase_sampling', 'subset')
