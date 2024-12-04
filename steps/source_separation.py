import pandas as pd
import torch
from typing_extensions import Annotated
from zenml import log_artifact_metadata, step
from zenml.config.resource_settings import ResourceSettings

from thesis_csi.logging import get_logger
from thesis_csi.source_separation.demucs import run_source_separation

logger = get_logger(__name__)


@step(
    settings={
        "resources": ResourceSettings(cpu_count=8, gpu_count=1, memory="8GiB"),
        # "orchestrator.vertex": VertexOrchestratorSettings(
        #     node_selector_constraint=("cloud.google.com/gke-accelerator", "NVIDIA_L4")
        # ),
    },
)
def source_separation(
    df: pd.DataFrame,
    model_name: str,
    bucket_name: str,  # TODO: Remove from parameters and add to config
    gs_mount_path: str = "/gcs/",  # TODO: Remove from parameters
    source: str = "",
) -> Annotated[pd.DataFrame, "df"]:
    """Source separation pipeline step.

    Args:
        dataset (Dataset): Dataset
        separator (str): Separator
        bucket_name (str): Bucket name
        start_index (int, optional): Start index. Defaults to None.

    Returns:
        Dataset: Dataset

    """
    torch.multiprocessing.set_start_method("spawn")

    logger.info(f"Source separation with {model_name}...")

    if "demucs" in model_name:
        df, stems_list = run_source_separation(
            df, model_name, bucket_name, gs_mount_path, source=source
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    meta_data = {f"{stem}_count": int(df[f"{stem}_{model_name}"].count()) for stem in stems_list}
    meta_data["Source separation success"] = int(
        df[f"source_separation_status_{model_name}"].count()
    )
    meta_data["Rows"] = int(df.shape[0])

    log_artifact_metadata(
        metadata=meta_data,
    )
    return df
