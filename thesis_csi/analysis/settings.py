import dataclasses


@dataclasses.dataclass
class GridSettings:
    """Settings for grid lines."""

    enabled: bool = True
    linestyle: str = "--"
    linewidth: float = 0.5
    which: str = "both"


FIG_SIZE = (10, 5)
GRID_SETTINGS = GridSettings()
