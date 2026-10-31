"""Execution-only options; no changes to targets or sampling distribution."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


DEFAULTS = {
    "selective_logits": True,
    "fast_prefix": True,
    "language_checkpointing": False,
    "vision_checkpointing": False,
    "decoder_checkpointing": False,
    "refinement_chunk_size": 0,
    "fused_adamw": "auto",
    "prefetch_batches": 0,
}


def prefetch(items, prepare, depth=0):
    """Bounded, ordered CPU preparation with exceptions delivered to the caller.

    One worker avoids processor reentrancy and does not touch CUDA or global RNG.
    No sampling ahead changes the main process's training RNG state.
    """
    if not depth:
        for item in items:
            yield prepare(item)
        return
    items = iter(items)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-input") as pool:
        pending = deque()
        for _ in range(depth):
            item = next(items, None)
            if item is None:
                break
            pending.append(pool.submit(prepare, item))
        while pending:
            value = pending.popleft().result()
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(prepare, item))
            yield value
