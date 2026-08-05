from __future__ import annotations

from typing import NamedTuple


class ConvLayer(NamedTuple):
    """Describe one temporal convolution layer.

    Height parameters operate on time and determine the consumed left context. Width
    parameters operate on the embedding axis and are expected to preserve that width;
    ``TemporalCNN`` validates these constraints when constructing the network.
    """

    out_channels: int
    kernel_height: int
    kernel_width: int
    pad_height: int
    pad_width: int
