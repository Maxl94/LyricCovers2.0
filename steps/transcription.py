from typing import Optional

import pandas as pd
import torch
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.config.resource_settings import ResourceSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import VertexOrchestratorSettings

from thesis_csi.enum import Features
from thesis_csi.logging import get_logger
from thesis_csi.shared.utils import update_model_tags
from thesis_csi.transcription.whisper import (
    run_transcription_process,
)

logger = get_logger(__name__)


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, gpu_count=1, memory="8GiB"),
        "orchestrator.vertex": VertexOrchestratorSettings(
            node_selector_constraint=("cloud.google.com/gke-accelerator", "NVIDIA_L4")
        ),
    },
)
def transcribe(
    df: pd.DataFrame,
    model_name: str,
    source_separation_config: Optional[dict] = None,
) -> Annotated[pd.DataFrame, "df_transcription"]:
    """Transcribe vocals to text.

    Args:
        df (pd.DataFrame): DataFrame
        bucket_name (str): Bucket name
        model_name (str): Model name.
        sample_rate (int, optional): Sample rate. Defaults to 16000.

    Returns:
        pd.DataFrame: DataFrame
    """
    torch.multiprocessing.set_start_method("spawn")

    logger.info("Adding transcription column to DataFrame")

    tags = [model_name]
    ss_model = source_separation_config.get("c_vocals", "").replace("vocals_", "")

    if ss_model != "":
        logger.info(f"Separating vocals using {ss_model}")
        tags.append(ss_model)

    update_model_tags(tags)

    logger.info("Transcribing vocals to text")
    df = run_transcription_process(
        df=df,
        model_name=model_name,
        source_separation_config=source_separation_config,
    )

    log_artifact_metadata(
        artifact_name="df_transcription",
        metadata={
            "transcription_count": int(df[Features.TRANSCRIPTION.column].count()),
            "model": model_name,
        },
    )

    return df
