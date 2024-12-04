import pandas as pd
from matplotlib import pyplot as plt

from thesis_csi.analysis import settings


def number_of_covers_original(df: pd.DataFrame, bins: int = 200) -> tuple[plt.Figure, pd.Series]:
    """Creates a plot and stats of number of covers per original song.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        tuple[plt.Figure, pd.Series]: Matplotlib figure and pandas Series of stats
    """

    df_org_grouped = df.groupby("original_id")["id"].count()
    stats = {
        "mean": df_org_grouped.mean(),
        "median": df_org_grouped.median(),
        "std": df_org_grouped.std(),
        "min": df_org_grouped.min(),
        "max": df_org_grouped.max(),
        "top_1": df[df["original_id"].isin(df_org_grouped.nlargest(1).index) & ~df["is_cover"]][
            "title"
        ].values[0],
        "top_1_count": df_org_grouped.nlargest(1).values[0],
    }

    fig, ax = plt.subplots(figsize=settings.FIG_SIZE)
    ax.hist(df_org_grouped, bins=bins, log=True)
    ax.grid(
        settings.GRID_SETTINGS.enabled,
        which=settings.GRID_SETTINGS.which,
        linestyle=settings.GRID_SETTINGS.linestyle,
        linewidth=settings.GRID_SETTINGS.linewidth,
    )
    ax.set_title("Number of covers per original song")
    ax.set_xlabel("Number of covers")
    ax.set_ylabel("Number of songs")

    return fig, stats
