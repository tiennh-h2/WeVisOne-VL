"""Deterministic shuffled epochs; padded distributed slots have zero loss weight."""
from bisect import bisect_right
import math
import random


class EpochSchedule:
    def __init__(self, lengths, epochs, world_size, accumulation, seed):
        if not lengths or any(n <= 0 for n in lengths):
            raise ValueError('epoch sources must be nonempty')
        self.ends = []
        for length in lengths:
            self.ends.append(length + (self.ends[-1] if self.ends else 0))
        self.size = self.ends[-1]
        self.world_size, self.accumulation, self.seed = world_size, accumulation, seed
        self.steps_per_epoch = math.ceil(self.size / (world_size * accumulation))
        self.max_steps = self.steps_per_epoch * epochs
        self._epoch, self._order = None, None

    def record(self, step, micro, rank):
        epoch, step_in_epoch = divmod(step, self.steps_per_epoch)
        if epoch != self._epoch:
            self._order = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(self._order)
            self._epoch = epoch
        slot = (step_in_epoch * self.accumulation + micro) * self.world_size + rank
        index = self._order[slot % self.size]
        source = bisect_right(self.ends, index)
        return source, index - (self.ends[source-1] if source else 0), epoch, slot < self.size
