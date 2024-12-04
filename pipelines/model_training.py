from typing import Optional, Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import Model, pipeline
from zenml.client import Client

from steps.evaluation import evaluate
from steps.features import prepare_features, train_test_split
from steps.training import train
from steps.transcription import transcribe
from thesis_csi.model.model import LyricsModel


@pipeline(
    name="thesis-model-training",
    model=Model(
        name="ThesisEmbeddingModel",
        description="Embedding for lyrics",
        delete_new_version_on_failure=True,
    ),
)
def model_training(
    dataset_id: str,
    source_separation_config: Optional[dict] = None,
    transcription_model: Optional[str] = "openai/whisper-tiny",
    split_ids: Optional[str] = None,
    transcription_id: Optional[str] = None,
) -> Tuple[Annotated[LyricsModel, "model"], Annotated[pd.DataFrame, "evaluation"]]:
    """Model training pipeline."""

    if source_separation_config is None:
        source_separation_config = {
            "c_status": "youtube_download_status",
            "c_vocals": "youtube_download_gs_path",
        }

    client = Client()

    if transcription_id is not None:
        df_transcription = client.get_artifact_version(transcription_id).load()
    else:
        df = client.get_artifact_version(dataset_id)

        df_transcription = transcribe(
            df=df,
            model_name=transcription_model,
            source_separation_config=source_separation_config,
        )

    df_features, n_labels = prepare_features(df_transcription)

    if split_ids is None:
        df_train, df_val, df_test = train_test_split(df_features)
    else:
        split_ids_artifact = client.get_artifact_version(split_ids).load()
        df_train, df_val, df_test = train_test_split(df_features, split_ids=split_ids_artifact)

    best_model = train(df_train, df_val)

    return evaluate(best_model, df_test)
