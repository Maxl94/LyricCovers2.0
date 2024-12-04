import numpy as np
import pandas as pd
from zenml import log_artifact_metadata, save_artifact

from thesis_csi.enum import Labels
from thesis_csi.logging import get_logger

logger = get_logger(__name__)


def create_song_ids_split(df: pd.DataFrame = None, splits: tuple = None) -> dict:
    """Create song IDs split.

    Args:
        df (pd.DataFrame, optional): DataFrame. Defaults to "gs://<google-cloud-project-id>-data/data/df_spleeter.parquet".
        splits (tuple, optional): Splits. Defaults to (0.6, 0.2, 0.2).

    Returns:
        dict: Split IDs
    """

    if df is None:
        logger.info("Loading DataFrame")
        df = pd.read_parquet("gs://<google-cloud-project-id>-data/data/df_spleeter.parquet")

    song_ids = df[Labels.ORIGINAL_SONG_ID.column].unique()

    if splits is None:
        logger.info("Using default splits")
        splits = (0.6, 0.2, 0.2)

    np.random.seed(42)
    np.random.shuffle(song_ids)

    train_ids = song_ids[: int(len(song_ids) * splits[0])]
    val_ids = song_ids[int(len(song_ids) * splits[0]) : int(len(song_ids) * (splits[0] + splits[1]))]
    test_ids = song_ids[int(len(song_ids) * (splits[0] + splits[1])) :]

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    logger.info("Saving split IDs")
    save_artifact(split_ids, "split_ids")

    log_artifact_metadata(
        artifact_name="split_ids",
        metadata={
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    )

    logger.info("Split IDs saved")
    return split_ids


if __name__ == "__main__":
    create_song_ids_split()
