from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Layout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def offset(self, coordinate: Sequence[int]) -> int:
        if len(coordinate) != len(self.shape):
            raise ValueError("coordinate rank does not match layout rank")
        return sum(index * step for index, step in zip(coordinate, self.stride))


def make_row_major_layout(shape: Iterable[int]) -> Layout:
    extents = tuple(int(value) for value in shape)
    if not extents or any(value <= 0 for value in extents):
        raise ValueError("shape must contain positive extents")

    strides = [1] * len(extents)
    for axis in range(len(extents) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * extents[axis + 1]

    return Layout(shape=extents, stride=tuple(strides))


def linear_size(layout: Layout) -> int:
    return prod(layout.shape)
