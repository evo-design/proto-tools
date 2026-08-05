"""proto_tools/tools/molecular_docking/vina/helpers.py.

Notebook helpers for inspecting a docking request before it is dispatched.
"""

import logging
import math
from typing import Any

from proto_tools.tools.molecular_docking.vina.vina_docking import (
    VinaDockingConfig,
    VinaDockingInput,
    VinaReferenceLigandBox,
)

logger = logging.getLogger(__name__)

_MAX_GRID_POINTS = 2_000_000

# Corner pairs of a unit cube, as (-1, +1) offsets per axis, that share exactly two
# coordinates and so span one of its twelve edges.
_UNIT_CUBE_CORNERS = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
_UNIT_CUBE_EDGES = [
    (a, b)
    for i, a in enumerate(_UNIT_CUBE_CORNERS)
    for b in _UNIT_CUBE_CORNERS[i + 1 :]
    if sum(1 for axis in range(3) if a[axis] != b[axis]) == 1
]


def _draw_box_edges(
    viewer: Any,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    radius: float,
    color: str = "magenta",
) -> None:
    """Draw the twelve edges of a search box as cylinders of the given radius."""

    def corner(offsets: tuple[int, int, int]) -> dict[str, float]:
        return {
            axis_name: center[axis] + offsets[axis] * size[axis] / 2.0 for axis, axis_name in enumerate(("x", "y", "z"))
        }

    for start, end in _UNIT_CUBE_EDGES:
        viewer.addCylinder(
            {
                "start": corner(start),
                "end": corner(end),
                "radius": radius,
                "color": color,
                "fromCap": 1,
                "toCap": 1,
            }
        )


def visualize_search_box(
    inputs: VinaDockingInput,
    config: VinaDockingConfig | None = None,
    width: int = 900,
    height: int = 600,
    edge_radius: float = 0.15,
) -> None:
    """Show the receptor with the resolved search box drawn as a wireframe cage.

    A misplaced or undersized box is the most common way a docking run goes wrong, and
    neither failure is visible from the coordinates alone. Passing the same ``inputs`` and
    ``config`` the run will use shows the box Vina will actually search, including the
    coordinates resolved from a :class:`VinaReferenceLigandBox`.

    When the box was derived from a reference ligand, that ligand is drawn too: it carries
    real coordinates in the receptor frame, so it shows whether the cage sits on the pocket.

    Args:
        inputs (VinaDockingInput): The docking request to inspect.
        config (VinaDockingConfig | None): Config for the run. When given, the affinity-map
            grid spacing and point count are reported, and an oversized grid is flagged
            before dispatch rather than at run time.
        width (int): Viewer width in pixels.
        height (int): Viewer height in pixels.
        edge_radius (float): Radius of the box edges in angstroms. WebGL clamps line
            widths to a single pixel, so the cage is built from cylinders to stay legible
            against the receptor.
    """
    import py3Dmol

    search_box = inputs.resolved_search_box()
    center, size = search_box.center, search_box.size

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(inputs.receptor.structure_pdb, "pdb")
    viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum"}})
    _draw_box_edges(viewer, center, size, edge_radius)

    if isinstance(inputs.search_box, VinaReferenceLigandBox):
        viewer.addModel(inputs.search_box.reference_ligand.structure_pdb, "pdb")
        viewer.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon"}})

    if config is not None:
        # Mirrors _validated_grid_point_count so the reported count matches the one
        # the tool enforces at dispatch.
        dimensions = tuple(math.ceil(axis / config.grid_spacing) + 1 for axis in size)
        grid_points = dimensions[0] * dimensions[1] * dimensions[2]
        logger.info(
            "Affinity-map grid: %d x %d x %d = %s points at %.3f A spacing",
            *dimensions,
            f"{grid_points:,}",
            config.grid_spacing,
        )
        if grid_points > _MAX_GRID_POINTS:
            logger.warning(
                "Grid would need %s points, above the %s maximum; widen grid_spacing or "
                "shrink the search box before dispatching",
                f"{grid_points:,}",
                f"{_MAX_GRID_POINTS:,}",
            )

    viewer.zoomTo()
    viewer.show()
