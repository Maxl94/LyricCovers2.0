from typing import Optional, Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step

from thesis_csi.enum import Features, Labels, Metadata
from thesis_csi.logging import get_logger

logger = get_logger(__name__)


@step
def prepare_features(
    df: pd.DataFrame,
) -> Tuple[Annotated[pd.DataFrame, "dataset"], Annotated[int, "n_labels"]]:
    """Prepare features.

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with features
    """
    logger.info("Preparing features")

    columns_metadata = [Metadata.SONG_ID.column, Metadata.IS_COVER.column]

    logger.info("Selecting transcriptions")
    df_transcription = df.loc[
        df[Features.TRANSCRIPTION.column].notna(),
        [Features.TRANSCRIPTION.column, Labels.ORIGINAL_SONG_ID.column] + columns_metadata,
    ]
    df_transcription = df_transcription.rename(columns={Features.TRANSCRIPTION.column: Features.SONG_TEXT.column})
    df_transcription[Metadata.SONG_TEXT_TYPE.column] = Features.TRANSCRIPTION.column

    logger.info("Selecting lyrics")
    df_lyrics = df.loc[
        df[Features.LYRICS.column].notna(),
        [Features.LYRICS.column, Labels.ORIGINAL_SONG_ID.column] + columns_metadata,
    ]
    df_lyrics = df_lyrics.rename(columns={Features.LYRICS.column: Features.SONG_TEXT.column})
    df_lyrics[Metadata.SONG_TEXT_TYPE.column] = Features.LYRICS.column

    logger.info("Concatenating features")
    df_features = pd.concat([df_transcription, df_lyrics])

    logger.info("Map labels")
    label_map = {label: i for i, label in enumerate(df_features[Labels.ORIGINAL_SONG_ID.column].unique())}
    df_features[Labels.LABEL.column] = df_features[Labels.ORIGINAL_SONG_ID.column].map(label_map)
    n_labels = len(label_map)

    log_artifact_metadata(
        artifact_name="dataset",
        metadata={
            "num_transcriptions": len(df_transcription),
            "num_lyrics": len(df_lyrics),
            "original_songs": len(df_features[Labels.ORIGINAL_SONG_ID.column].unique()),
        },
    )

    return df_features, n_labels


@step
def train_test_split(
    df: pd.DataFrame,
    splits: Optional[Tuple[int, int, int]] = (60, 20, 20),
    split_ids: Optional[dict] = None,
) -> Tuple[
    Annotated[pd.DataFrame, "train"],
    Annotated[pd.DataFrame, "val"],
    Annotated[pd.DataFrame, "test"],
]:
    unique_songs = df[~df[Metadata.IS_COVER.column]][Labels.ORIGINAL_SONG_ID.column].unique()

    if split_ids is None:
        train_size = int(len(unique_songs) * splits[0] / 100)
        val_size = int(len(unique_songs) * splits[1] / 100)
        test_size = int(len(unique_songs) * splits[2] / 100)

        logger.info(
            f"Splitting dataset original songs into train ({train_size})"
            f", val ({val_size})"
            f", and test ({test_size})"
        )

        train_songs = unique_songs[:train_size]
        val_songs = unique_songs[train_size : train_size + val_size]
        test_songs = unique_songs[train_size + val_size :]
    else:
        logger.info("Using provided split IDs")
        train_songs = split_ids["train"]
        val_songs = split_ids["val"]
        test_songs = split_ids["test"]

    df_train = df[df[Labels.ORIGINAL_SONG_ID.column].isin(train_songs)]
    df_val = df[df[Labels.ORIGINAL_SONG_ID.column].isin(val_songs)]
    df_test = df[df[Labels.ORIGINAL_SONG_ID.column].isin(test_songs)]

    def log_metadata(df: pd.DataFrame, name: str):
        original_songs = len(df[Labels.ORIGINAL_SONG_ID.column].unique())
        unique_songs = len(df[Metadata.SONG_ID.column].unique())
        cover_songs = len(df[df[Metadata.IS_COVER.column]])
        cover_songs_unique = len(df[df[Metadata.IS_COVER.column]][Metadata.SONG_ID.column].unique())

        logger.info(f"Original songs in {name}: {original_songs}")
        logger.info(f"Unique songs in {name}: {unique_songs}")
        logger.info(f"Cover songs in {name}: {cover_songs}")
        logger.info(f"Cover songs unique in {name}: {cover_songs_unique}")

        log_artifact_metadata(
            artifact_name=name,
            metadata={
                "original_songs": original_songs,
                "unique_songs": unique_songs,
                "cover_songs": cover_songs,
                "cover_songs_unique": cover_songs_unique,
            },
        )

    log_metadata(df_train, "train")
    log_metadata(df_val, "val")
    log_metadata(df_test, "test")

    return df_train, df_val, df_test
