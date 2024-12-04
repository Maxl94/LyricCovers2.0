import time
from typing import Optional

import pandas as pd
import torch
import torchaudio
from transformers import pipeline
from zenml import get_step_context, save_artifact

from thesis_csi.enum import Features, Metadata
from thesis_csi.logging import get_logger, progress_bar
from thesis_csi.shared.utils import get_device

logger = get_logger(__name__)


def run_transcription_process(
    df: pd.DataFrame,
    model_name: str,
    source_separation_config: Optional[dict] = None,
) -> pd.DataFrame:
    """Run transcription process.

    Args:
        df (pd.DataFrame): DataFrame
        bucket_name (str): Bucket name
        model_name (str): Model name
        sample_rate (int): Sample rate
        gs_mount_path (str): Mount path

    Returns:
        pd.DataFrame: DataFrame
    """
    start_time = time.time()

    device = get_device()
    logger.info(f"Using device {device}")

    logger.info(f"Loading model '{model_name}'")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device,
        chunk_length_s=30,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Starting transcription...")

    if source_separation_config is None:
        source_separation_config = {
            "c_status": Metadata.SOURCE_SEPARATION_STATUS.column,
            "c_vocals": Features.VOCALS.column,
        }

    # remove all this code and use batch_size for pipeline
    # https://huggingface.co/openai/whisper-large-v3#chunked-long-form
    idx = 0
    interval = 1000
    df_success = df[df[source_separation_config["c_status"]] == "success"]

    for row in progress_bar(
        df_success[["id", source_separation_config["c_vocals"], "language"]].itertuples(index=False),
        message="Transcribing",
        interval=interval // 10,
        logger=logger,
        total_steps=len(df_success),
    ):
        song_id, song_path, language = row

        try:
            wave, sr = torchaudio.load(song_path[0].replace("gs://", "/gcs/"))
            if sr != 16000:
                wave = torchaudio.transforms.Resample(sr, 16000)(wave).numpy()

            # make monitonic
            wave = wave.mean(axis=0)

            transcription = pipe(wave, batch_size=32)["text"]
            status = "success"
            logger.info(f"Transcription for {song_path}: {transcription}")
        except Exception as e:
            logger.error(f"Error processing {song_path}: {e}", exc_info=True)
            status = "error"
            transcription = None
        finally:
            df.loc[df["id"] == song_id, Metadata.TRANSCRIPTION_STATUS.column] = status
            df.loc[df["id"] == song_id, Features.TRANSCRIPTION.column] = transcription

        if idx % interval == 0:
            version = get_step_context().model.version + "_checkpoint_" + str(idx)
            save_artifact(df.copy(), "df_transcription_checkpoint", version=version)
            logger.info(f"Saved checkpoint {idx}, version {version}")
        idx += 1

    logger.info(f"Transcription process took {time.time() - start_time:.0f} seconds")
    version = get_step_context().model.version + "_checkpoint_final"
    save_artifact(df.copy(), "df_transcription_checkpoint", version=version)
    return df
