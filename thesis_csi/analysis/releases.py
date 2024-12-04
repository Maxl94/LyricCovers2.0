from typing import Literal

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange

from thesis_csi.analysis import settings
from thesis_csi.logging import get_logger
from thesis_csi.shared.utils import authenticate_with_service_account

logger = get_logger(__name__)


def get_release_years(df: pd.DataFrame) -> pd.Series:
    from google.cloud.firestore import Client

    client = Client(
        project="thesis-mb-csi-dev-f027",
        credentials=authenticate_with_service_account(
            "/gcs/<google-cloud-project-id>-data/thesis-mb-csi-dev-f027-ea4067cc2bdc.json"
        ),
    )
    collection = client.collection("song")

    # Get unique song IDs
    song_ids = df["id"].unique()

    # Batch get documents (Firestore allows max 500 per batch)
    batch_size = 30
    release_dates = {}

    for i in trange(0, len(song_ids), batch_size, desc="Fetching release dates"):
        batch = song_ids[i : i + batch_size]
        docs = collection.where("__name__", "in", [str(id) for id in batch]).stream()

        for doc in docs:
            try:
                data = doc.to_dict()
                release_date = data["response"]["response"]["song"]["release_date"]
                release_dates[int(doc.id)] = release_date
            except (KeyError, TypeError):
                release_dates[int(doc.id)] = None

    # Vectorized date processing
    df["release_date"] = df["id"].map(release_dates)
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d", errors="coerce")

    return df["release_date"]


def releases_per_year(df: pd.DataFrame, song_type: Literal["original", "cover"]) -> tuple[plt.Figure, pd.Series]:
    """Creates a plot and stats of release years for original or cover songs.

    Args:
        df (pd.DataFrame): DataFrame containing release years
        song_type (Literal["original", "cover"]): Type of song to plot

    Returns:
        tuple[plt.Figure, pd.Series]: Matplotlib figure and pandas Series of stats
    """
    if "release_date" not in df.columns:
        logger.warning("No release years found in DataFrame. Getting from Firestore.")
        df["release_date"] = get_release_years(df)

    df["release_year"] = df["release_date"].dt.year

    if song_type == "original":
        data = df[~df["is_cover"]]
    else:
        data = df[df["is_cover"]]

    fig, ax = plt.subplots(figsize=settings.FIG_SIZE)
    data["release_year"].hist(bins=range(1950, 2021), ax=ax)
    ax.grid(
        settings.GRID_SETTINGS.enabled,
        which=settings.GRID_SETTINGS.which,
        linestyle=settings.GRID_SETTINGS.linestyle,
        linewidth=settings.GRID_SETTINGS.linewidth,
    )
    ax.set_title(f"Release years of {song_type} songs")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of songs")

    stats = data["release_year"].describe()

    return fig, stats
