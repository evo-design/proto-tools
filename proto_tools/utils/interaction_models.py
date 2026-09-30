"""Shared data and output models for sequence interaction probability maps."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, model_validator

from proto_tools.utils.tool_io import BaseToolOutput

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Plot styling follows the Minerva publication figures (github.com/garykbrixi/minerva, Apache-2.0).
CHANNEL_COLORS = {"base_pairing": "#DA3832", "repeat": "#5F308C", "protein": "#7DB4E0"}
CHANNEL_LABELS = {"base_pairing": "Base pairing", "repeat": "Repeats", "protein": "Protein contacts"}
POSITION_COLORS = {"protein": CHANNEL_COLORS["protein"], "nucleotide": "#D9D6CF", "marker": "#4A4A4A"}
# Base pairing wins a cell wherever it reaches this value, so faint RNA structure stays visible.
BASE_PAIRING_OVERRIDE = 0.5


class SequenceInteractionMap(BaseModel):
    """A dense square probability matrix with labeled axes.

    Labels may represent residues, bases, model tokens, or numbered positions.
    Values need not be symmetric or have a zero diagonal.

    Attributes:
        axis_labels (list[str]): Ordered labels shared by both matrix axes.
        values (list[list[float]]): Square matrix of finite interaction probabilities.
    """

    axis_labels: list[str] = Field(
        title="Axis Labels", description="Ordered labels shared by both matrix axes", min_length=1
    )
    values: list[list[float]] = Field(
        title="Values", description="Square interaction probability matrix over the labeled axes"
    )

    @model_validator(mode="after")
    def validate_matrix(self) -> "SequenceInteractionMap":
        """Validate square shape, axis alignment, and finite probability values."""
        length = len(self.axis_labels)
        if len(self.values) != length or any(len(row) != length for row in self.values):
            raise ValueError("Interaction matrix must be square and match its axis labels")
        if any(not 0.0 <= value <= 1.0 for row in self.values for value in row):
            raise ValueError("Interaction probabilities must be finite values in [0, 1]")
        return self


class SequenceInteractions(BaseModel):
    """Named interaction channels for one input sequence.

    Attributes:
        maps (dict[str, SequenceInteractionMap]): Named interaction probability maps.
    """

    maps: dict[str, SequenceInteractionMap] = Field(title="Maps", description="Named interaction probability maps")


class SequenceInteractionsOutput(BaseToolOutput):
    """Dense interaction maps, one bundle per input sequence.

    Attributes:
        results (list[SequenceInteractions]): Interaction bundles in input order.
    """

    results: list[SequenceInteractions] = Field(title="Results", description="Interaction bundles in input order")

    @property
    def output_format_options(self) -> list[str]:
        """Return supported export formats."""
        return ["json", "npz"]

    @property
    def output_format_default(self) -> str:
        """Prefer compact archives for dense interaction matrices."""
        return "npz"

    def _export_output(self, export_path: str | Path, file_format: str) -> None:
        """Export matrices and axis labels without pickle-dependent object arrays.

        Raises:
            ValueError: If channel names produce duplicate NPZ keys or the format is unsupported.
        """
        if file_format == "json":
            Path(str(export_path) + ".json").write_text(json.dumps([r.model_dump() for r in self.results]))
        elif file_format == "npz":
            import numpy as np

            arrays: dict[str, Any] = {}
            for index, result in enumerate(self.results):
                for channel, interaction_map in result.maps.items():
                    values_key = f"{index}_{channel}"
                    labels_key = f"{index}_{channel}_axis_labels"
                    for key in (values_key, labels_key):
                        if key in arrays:
                            raise ValueError(f"Interaction channel names produce duplicate NPZ key: {key}")
                    arrays[values_key] = np.asarray(interaction_map.values, dtype=np.float32)
                    arrays[labels_key] = np.asarray(interaction_map.axis_labels, dtype=str)
            np.savez_compressed(str(export_path) + ".npz", **arrays)
        else:
            raise ValueError(f"Unsupported format: {file_format}")

    def plot(
        self,
        index: int = 0,
        *,
        channel: str | None = None,
        window: tuple[int, int] | None = None,
        mask_by_modality: bool = True,
        show_track: bool = True,
        ax: "Axes | None" = None,
        figsize: tuple[float, float] = (6.0, 6.0),
    ) -> "Figure":
        """Plot one sequence's interaction maps as a heatmap.

        By default the base-pairing, repeat, and protein-contact channels are overlaid in one
        panel: each cell takes the color of its strongest channel, faded toward white as the
        probability drops, and base pairing wins wherever it reaches 0.5. Pass ``channel`` to
        plot a single map with a colorbar instead.

        Args:
            index (int): Position of the sequence in ``results``.
            channel (str | None): Plot only this channel; ``None`` overlays the known channels.
            window (tuple[int, int] | None): 1-indexed inclusive ``(start, end)`` positions to zoom into.
            mask_by_modality (bool): Show protein contacts only between amino acids, and base pairing
                and repeats only between nucleotides, using uppercase/lowercase axis labels.
            show_track (bool): Draw a strip along the top and left edges marking protein, nucleotide,
                and strand-marker positions.
            ax (Axes | None): Axes to draw into; a new figure is created when omitted.
            figsize (tuple[float, float]): Size of a newly created figure.

        Returns:
            Figure: The matplotlib figure containing the plot.

        Raises:
            ValueError: If ``index``, ``channel``, or ``window`` does not match the results.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if not 0 <= index < len(self.results):
            raise ValueError(f"index {index} is out of range for {len(self.results)} result(s)")
        maps = self.results[index].maps
        if channel is not None and channel not in maps:
            raise ValueError(f"Unknown channel {channel!r}; available: {sorted(maps)}")
        names = [channel] if channel is not None else [name for name in CHANNEL_COLORS if name in maps]
        if not names:
            raise ValueError(f"No known channel to overlay in {sorted(maps)}; pass channel= to plot one map")
        length = len(maps[names[0]].axis_labels)
        start, end = window if window is not None else (1, length)
        if not 1 <= start <= end <= length:
            raise ValueError(f"window must satisfy 1 <= start <= end <= {length}; got {window}")

        # Crop every map to the window and classify its positions from the axis labels.
        crop = slice(start - 1, end)
        position_types = _position_types(maps[names[0]].axis_labels[crop])
        values = {name: np.asarray(maps[name].values)[crop, crop] for name in names}
        if mask_by_modality:
            values = _mask_by_modality(values, position_types)

        fig, ax = plt.subplots(figsize=figsize) if ax is None else (cast("Figure", ax.figure), ax)
        if channel is None:
            _draw_overlay(ax, values)
        else:
            _draw_channel(fig, ax, channel, values[channel])

        # Label ticks with 1-indexed sequence positions.
        ticks = np.arange(0, end - start + 1, max(1, (end - start + 1) // 4))
        ax.set_xticks(ticks, [str(tick + start) for tick in ticks])
        ax.set_yticks(ticks, [str(tick + start) for tick in ticks])
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        if show_track:
            _draw_track(ax, position_types)
        ax.set_title("Interaction maps" if channel is None else CHANNEL_LABELS.get(channel, channel), pad=20)
        return fig


def _position_types(labels: list[str]) -> "np.ndarray":
    """Classify axis labels as protein (uppercase), nucleotide (lowercase), or marker."""
    import numpy as np

    return np.array(
        ["protein" if label[:1].isupper() else "nucleotide" if label[:1].islower() else "marker" for label in labels]
    )


def _mask_by_modality(values: dict[str, "np.ndarray"], position_types: "np.ndarray") -> dict[str, "np.ndarray"]:
    """Zero protein contacts outside amino-acid pairs and nucleotide channels outside nucleotide pairs."""
    import numpy as np

    def pairs(kind: str) -> "np.ndarray":
        return np.outer(position_types == kind, position_types == kind)

    allowed = {"protein": pairs("protein"), "base_pairing": pairs("nucleotide"), "repeat": pairs("nucleotide")}
    return {
        name: np.where(allowed[name], matrix, 0.0) if name in allowed else matrix for name, matrix in values.items()
    }


def _draw_overlay(ax: "Axes", values: dict[str, "np.ndarray"]) -> None:
    """Color each cell by its strongest channel, blended from white by that channel's value."""
    import numpy as np
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    names = list(values)
    stack = np.stack([values[name] for name in names])
    winner = np.argmax(stack, axis=0)
    if "base_pairing" in names:
        winner[values["base_pairing"] >= BASE_PAIRING_OVERRIDE] = names.index("base_pairing")
    strength = np.take_along_axis(stack, winner[None], axis=0)[0]
    colors = np.array([to_rgb(CHANNEL_COLORS[name]) for name in names])
    ax.imshow(1.0 - (1.0 - colors[winner]) * strength[..., None], interpolation="nearest")
    handles = [Patch(facecolor=CHANNEL_COLORS[name], label=CHANNEL_LABELS[name]) for name in names]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)


def _draw_channel(fig: "Figure", ax: "Axes", channel: str, values: "np.ndarray") -> None:
    """Draw one channel as a white-to-color heatmap with a probability colorbar."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(channel, ["#FFFFFF", CHANNEL_COLORS.get(channel, "#000000")])
    image = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Probability")


def _draw_track(ax: "Axes", position_types: "np.ndarray") -> None:
    """Draw position-type strips above and left of the heatmap."""
    import numpy as np
    from matplotlib.colors import to_rgb

    strip = np.array([to_rgb(POSITION_COLORS[kind]) for kind in position_types])
    for bounds, image in (((0.0, 1.01, 1.0, 0.03), strip[None]), ((-0.04, 0.0, 0.03, 1.0), strip[:, None])):
        strip_ax = ax.inset_axes(bounds)
        strip_ax.imshow(image, aspect="auto", interpolation="nearest")
        strip_ax.set_axis_off()
    # Keep the y tick labels clear of the left strip.
    ax.tick_params(axis="y", pad=22)
