import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from typing_extensions import Annotated
from zenml import log_artifact_metadata, log_model_metadata, step
from zenml.config.resource_settings import ResourceSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import VertexOrchestratorSettings

from thesis_csi.enum import Labels, Metadata
from thesis_csi.logging import get_logger
from thesis_csi.model.dataset import EmbeddingDataModule
from thesis_csi.model.metrics import (
    mean_average_precision,
    mean_rank_one,
    mean_reciprocal_rank,
    precision_at_k,
)
from thesis_csi.model.model import LyricsModel

logger = get_logger(__name__)


@step(
    settings={
        "resources": ResourceSettings(cpu_count=4, gpu_count=1, memory="64GiB"),
        "orchestrator.vertex": VertexOrchestratorSettings(
            node_selector_constraint=("cloud.google.com/gke-accelerator", "NVIDIA_L4")
        ),
    },
)
def evaluate(
    model: LyricsModel, df_test: pd.DataFrame, batch_size: int = 32
) -> Annotated[pd.DataFrame, "df_evaluation"]:
    logger.info("Evaluating model")

    logger.info("Preparing database and query dataframes")
    df_database = df_test[~df_test[Metadata.IS_COVER.column]]
    df_query = df_test[(df_test[Metadata.IS_COVER.column]) & (df_test.song_text_type == "transcription")]

    trainer = L.Trainer(
        accelerator="auto",
    )

    datamodule_database = EmbeddingDataModule(
        base_model=model.hparams.base_model_name,
        df_predict=df_database,
        batch_size=batch_size,
        max_length=model.hparams.max_length,
    )

    datamodule_query = EmbeddingDataModule(
        base_model=model.hparams.base_model_name,
        df_predict=df_query,
        batch_size=batch_size,
        max_length=model.hparams.max_length,
    )

    logger.info(f"Predicting embeddings for database {len(df_database)} samples")
    predictions_database = torch.concat(trainer.predict(model, datamodule=datamodule_database))

    logger.info(f"Predicting embeddings for query {len(df_query)} samples")
    predictions_query = torch.concat(trainer.predict(model, datamodule=datamodule_query))

    logger.info("Calculating nearest neighbors")
    nearest_neighbors = NearestNeighbors(n_neighbors=len(df_database), metric="euclidean", n_jobs=-1)
    nearest_neighbors.fit(predictions_database)

    distances, indices = nearest_neighbors.kneighbors(predictions_query)

    logger.info("Calculating metrics")
    relevant_documents = np.zeros((len(df_query), len(df_database)))

    for row_index, row in enumerate(indices):
        relevant_documents[row_index, :] = (
            df_database.iloc[row][Labels.ORIGINAL_SONG_ID.column]
            == df_query.iloc[row_index][Labels.ORIGINAL_SONG_ID.column]
        )

    mAP, mAP_std = mean_average_precision(relevant_documents, distances)
    logger.info(f"Mean Average Precision: {mAP:.4f} ± {mAP_std:.4f}")

    mrr, mrr_std, _ = mean_reciprocal_rank(relevant_documents, distances)
    logger.info(f"Mean Reciprocal Rank: {mrr:.4f} ± {mrr_std:.4f}")

    mr1, mr1_str, _ = mean_rank_one(relevant_documents, distances)
    logger.info(f"Mean Rank 1: {mr1:.4f} ± {mr1_str:.4f}")

    precision_at_1, precision_at_1_std = precision_at_k(relevant_documents, distances)
    logger.info(f"Precision at 1: {precision_at_1:.4f} ± {precision_at_1_std:.4f}")

    log_model_metadata(
        metadata={
            "mAP": float(mAP),
            "mAP_std": float(mAP_std),
            "mrr": float(mrr),
            "mrr_std": float(mrr_std),
            "mr1": float(mr1),
            "mr1_std": float(mr1_str),
            "precision_at_1": float(precision_at_1),
            "precision_at_1_std": float(precision_at_1_std),
        },
    )

    logger.info("Create evaluation dataframe")
    df_evaluation = pd.DataFrame(
        {
            "query_song_id": df_query[Metadata.SONG_ID.column].tolist(),
            "query_original_song_id": df_query[Labels.ORIGINAL_SONG_ID.column].tolist(),
            "predicted_song_id": df_database.iloc[indices[:, 0]][Metadata.SONG_ID.column].tolist(),
            "predicted_original_song_id": df_database.iloc[indices[:, 0]][Labels.ORIGINAL_SONG_ID.column].tolist(),
            "distance": distances[:, 0].tolist(),
        }
    )

    log_artifact_metadata(
        artifact_name="df_evaluation",
        metadata={
            "mAP": float(mAP),
            "mAP_std": float(mAP_std),
            "mrr": float(mrr),
            "mrr_std": float(mrr_std),
            "mr1": float(mr1),
            "mr1_std": float(mr1_str),
            "precision_at_1": float(precision_at_1),
            "precision_at_1_std": float(precision_at_1_std),
        },
    )

    return df_evaluation
