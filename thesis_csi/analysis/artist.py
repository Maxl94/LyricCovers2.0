from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from thesis_csi.analysis import settings


def covers_per_original_artist(
    df: pd.DataFrame, bins: int = 200, outliers_cut: Union[int, None] = 200
) -> tuple[plt.Figure, pd.Series]:
    """Creates a plot and stats of number of covers per original artist.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs
        bins (int): Number of bins for histogram, default 200. Set to None for no cutting.


    Returns:
        tuple[plt.Figure, pd.Series]: Matplotlib figure and pandas Series of stats
    """

    covers_df = df[df["is_cover"]]
    originals_df = df[~df["is_cover"]][["id", "artist", "artist_id"]]

    # Vectorized merge and aggregation
    merged_df = covers_df.merge(
        originals_df, left_on="original_id", right_on="id", suffixes=("_cover", "_original")
    )

    # Vectorized counting using value_counts()
    cover_counts = merged_df["artist_original"].value_counts()

    if outliers_cut:
        cover_counts_no_outliers = cover_counts[cover_counts < outliers_cut]
    else:
        cover_counts_no_outliers = cover_counts

    fig, ax = plt.subplots(figsize=settings.FIG_SIZE)
    ax.hist(cover_counts_no_outliers, bins=bins, log=True)
    ax.grid(
        settings.GRID_SETTINGS.enabled,
        which=settings.GRID_SETTINGS.which,
        linestyle=settings.GRID_SETTINGS.linestyle,
        linewidth=settings.GRID_SETTINGS.linewidth,
    )
    ax.set_title("Number of covers per original artist")
    ax.set_xlabel("Number of covers")
    ax.set_ylabel("Number of artists")

    stats = {
        "total_artists": len(cover_counts),
        "mean_covers": cover_counts.mean(),
        "median_covers": cover_counts.median(),
        "max_covers": cover_counts.max(),
        "top_10_artist_with_most_covers": cover_counts.head(10),
    }

    return fig, pd.Series(stats)


def get_unique_original_artists(df: pd.DataFrame) -> pd.Series:
    """Get number of unique original artists.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Number of unique original artists
    """

    return df[~df["is_cover"]]["artist"].nunique()


def get_avg_song_per_original_artist(df: pd.DataFrame) -> pd.Series:
    """Get average number of songs per original artist.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Average number of songs per original artist
    """

    return df[~df["is_cover"]].groupby("artist").size().mean()


def get_unique_cover_artists(df: pd.DataFrame) -> pd.Series:
    """Get number of unique cover artists.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Number of unique cover artists
    """

    return df[df["is_cover"]]["artist"].nunique()


def top_10_original_artist(df: pd.DataFrame) -> pd.Series:
    """Get top 10 original artists with most covers.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Top 10 original artists with most covers
    """

    df_org_artists = df[~df["is_cover"]][["id", "artist"]].groupby("artist").count()

    return df_org_artists.sort_values("id", ascending=False).head(10)["id"]


def top_10_cover_artist(df: pd.DataFrame) -> pd.Series:
    """Get top 10 cover artists with most covers.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Top 10 cover artists with most covers
    """

    df_cover_artists = df[df["is_cover"]][["id", "artist"]].groupby("artist").count()

    return df_cover_artists.sort_values("id", ascending=False).head(10)["id"]
