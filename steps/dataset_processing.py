from typing import Tuple

import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step

from thesis_csi.logging import get_logger

logger = get_logger(__name__)


@step
def spilt_dataset(
    gcs_path: str,
) -> Tuple[
    Annotated[pd.DataFrame, "split_0"],
    Annotated[pd.DataFrame, "split_1"],
    Annotated[pd.DataFrame, "split_2"],
    Annotated[pd.DataFrame, "split_3"],
]:
    """Split dataset into n equal parts. Either pass a DataFrame or a GCS path.

    Args:
        gcs_path (str): GCS path to the dataset.
        df (pd.DataFrame): DataFrame to split.

    Returns:
        list[pd.DataFrame]: List of DataFrames.
    """
    SPLITS = 4
    if gcs_path is not None:
        logger.info(f"Reading dataset from {gcs_path}")
        df = pd.read_parquet(gcs_path)
    else:
        logger.info("Using provided DataFrame")

    logger.info("Adding vocal_file column to DataFrame")
    df["vocal_file"] = None

    logger.info(f"Splitting dataset into {SPLITS} equal parts")
    splits = np.array_split(df, SPLITS)

    logger.info("Logging metadata for each split")
    for idx in range(4):
        log_artifact_metadata(
            artifact_name=f"split_{idx}",
            metadata={
                "split_sizes": len(splits[idx]),
            },
        )

    return splits[0], splits[1], splits[2], splits[3]


@step
def combine_datasets(
    df_part_0: pd.DataFrame,
    df_part_1: pd.DataFrame,
    df_part_2: pd.DataFrame,
    df_part_3: pd.DataFrame,
) -> pd.DataFrame:
    """Combine datasets into one.

    Args:
        df_part_0 (pd.DataFrame): DataFrame part 0
        df_part_1 (pd.DataFrame): DataFrame part 1
        df_part_2 (pd.DataFrame): DataFrame part 2
        df_part_3 (pd.DataFrame): DataFrame part 3

    Returns:
        pd.DataFrame: Combined DataFrame
    """
    logger.info("Combining datasets...")
    df = pd.concat([df_part_0, df_part_1, df_part_2, df_part_3])

    return df
