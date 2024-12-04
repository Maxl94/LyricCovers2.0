from typing import Optional

import pandas as pd
from zenml import pipeline
from zenml.client import Client

from steps.analysis import lyric_similarity
from steps.download import clean, crawl_tags, download_from_yt, get_release_year
from steps.importer import load_dataset
from steps.source_separation import source_separation
from thesis_csi.logging import get_logger

logger = get_logger(__name__)


@pipeline(
    name="thesis-feature-preparation",
)
def feature_preparation(
    gcs_path: Optional[str] = None,
    dataset_file_id: Optional[str] = None,
    source: Optional[str] = "genius",
) -> pd.DataFrame:
    """Feature preparation pipeline."""

    if gcs_path is None:
        df = Client().get_artifact_version(dataset_file_id).load()
    else:
        df = load_dataset(gcs_path=gcs_path)

    df = download_from_yt(df, source=source)
    df = clean(df)

    df = source_separation(df, source=source, model_name="htdemucs", bucket_name="<google-cloud-project-id>-data")

    df = crawl_tags(df)
    df = get_release_year(df)

    lyric_similarity(df)

    return df
