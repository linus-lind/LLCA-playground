from torch import Tensor, nn

from llca.models.modules.conv_layer import ConvLayer


class TemporalCNN(nn.Module):
    """Extract local temporal patterns while preserving the embedding width.

    Input ``x[N, T_in, D]`` is interpreted by 2-D convolutions as one channel with time
    as height and embedding width as width. Layer validation ensures width-preserving
    padding and prevents expansion of the time axis. After channel collapse, the output
    has shape ``[N, T_in - buffer_size, D]``. Callers prepend ``buffer_size`` historical
    rows when they require a specific output length.
    """

    def __init__(self, model_dim: int, layers: list[ConvLayer], *, dropout: float = 0.0) -> None:
        super().__init__()
        if not layers:
            raise ValueError("TemporalCNN requires at least one convolutional layer")
        self.model_dim = model_dim
        self.layers = [ConvLayer(*layer) for layer in layers]

        blocks: list[nn.Module] = []
        in_channels = 1
        for layer in self.layers:
            if 2 * layer.pad_width + 1 != layer.kernel_width:
                raise ValueError(
                    f"ConvLayer width padding {layer.pad_width} does not preserve model_dim for "
                    f"kernel_width {layer.kernel_width} (needs pad_width = (kernel_width - 1) / 2)"
                )
            if 2 * layer.pad_height > layer.kernel_height - 1:
                raise ValueError(
                    f"ConvLayer pad_height {layer.pad_height} would expand the time axis for "
                    f"kernel_height {layer.kernel_height}"
                )
            blocks.append(
                nn.Conv2d(
                    in_channels,
                    layer.out_channels,
                    kernel_size=(layer.kernel_height, layer.kernel_width),
                    padding=(layer.pad_height, layer.pad_width),
                )
            )
            blocks.append(nn.ELU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            in_channels = layer.out_channels
        self.convolutions = nn.Sequential(*blocks)
        self.collapse = nn.Conv2d(in_channels, 1, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        """Return the number of input timesteps influencing one output timestep."""
        return 1 + sum(layer.kernel_height - 1 for layer in self.layers)

    @property
    def buffer_size(self) -> int:
        """Return the input rows consumed by temporal shrinkage across all layers."""
        return sum(layer.kernel_height - 1 - 2 * layer.pad_height for layer in self.layers)

    def forward(self, x: Tensor) -> Tensor:
        """Convolve ``[N, T_in, D]`` into ``[N, T_in - buffer_size, D]``."""
        hidden = self.convolutions(x.unsqueeze(1))
        collapsed = self.collapse(hidden)
        out: Tensor = collapsed.squeeze(1)
        return out
