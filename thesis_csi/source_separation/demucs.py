import concurrent.futures
import os

import demucs.api
import pandas as pd
import torch
import torchaudio

from thesis_csi.logging import get_logger, progress_bar
from thesis_csi.shared.utils import get_device

logger = get_logger(__name__)


def process_song(
    song_id: str,
    file_path: str,
    separator: demucs.api.Separator,
    gs_mount_path: str,
    source: str,
    bucket_name: str,
):
    """Process song.

    Args:
        row (pd.Series): Row
        separator (demucs.api.Separator): Separator
        gs_mount_path (str): Mount path
        bucket_name (str): Bucket name
        device (str): Device

    Returns:
        Tuple[str, Dict[str, str], str]: Tuple with song id, stems and status

    """

    file_path = file_path.replace("gs://", gs_mount_path)
    stems = {}
    status = "success"
    model = "demucs"

    vocals_file_path = os.path.join("/gcs", bucket_name, source, model, "vocals", f"{song_id}.wav")

    with torch.no_grad():
        try:
            # ... (rest of the separation code per song) ...
            logger.info(f"Processing {vocals_file_path}...")
            if os.path.exists(vocals_file_path):
                logger.info(f"Vocals already separated for {file_path}. Using existing file.")
                stems["vocals"] = vocals_file_path.replace(gs_mount_path, "gs://")
            else:
                logger.info(f"Separating {file_path}...")

                wave, sr = torchaudio.load(file_path)
                if sr != 16000:
                    wave = torchaudio.transforms.Resample(sr, 16000)(wave)

                _, stems_data = separator.separate_tensor(wave)

                for stem, audio in stems_data.items():
                    stem_path = os.path.join(gs_mount_path, bucket_name, source, model, stem)

                    os.makedirs(stem_path, exist_ok=True)
                    bucket_file = os.path.join(stem_path, f"{song_id}.wav")

                    demucs.api.save_audio(audio, bucket_file, samplerate=separator.samplerate)

                    stems[stem] = bucket_file.replace(gs_mount_path, "gs://")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True, stack_info=True)
            status = "error"

    return song_id, stems, status


def run_source_separation(
    df: pd.DataFrame,
    model_name: str,
    bucket_name: str,
    gs_mount_path: str,
    source: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Run source separation.

    Args:
        df (pd.DataFrame): DataFrame
        model_name (str): Model name
        bucket_name (str): Bucket name
        gs_mount_path (str): Mount path

    Returns:
        pd.DataFrame: DataFrame
        list[str]: List of stems

    """
    c_status = f"source_separation_status_{model_name}"
    df[c_status] = None

    logger.info("Loading model...")

    separator = demucs.api.Separator(
        device=get_device(),
        model=model_name,
        jobs=os.cpu_count(),
    )

    logger.info(f"Using {os.cpu_count()} jobs for source separation.")

    logger.info("Starting source separation...")

    stems_list = ["vocals", "drums", "bass", "other"]
    # Vectorized column creation
    stem_columns = [f"{stem}_{model_name}" for stem in stems_list]
    df[stem_columns] = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Vectorized filtering and data preparation
        df_success = df[df["youtube_download_status"] == "success"]
        song_data = df_success[["id", "youtube_download_gs_path"]].values

        # Create futures using list comprehension
        futures = [
            executor.submit(
                process_song,
                song_id,
                song_path,
                separator,
                gs_mount_path,
                source,
                bucket_name,
            )
            for song_id, song_path in song_data
        ]

        # Process results using vectorized operations
        for future in progress_bar(
            concurrent.futures.as_completed(futures),
            message="Separating",
            interval=25,
            logger=logger,
        ):
            song_id, stems, status = future.result()
            # Vectorized updates
            mask = df["id"] == song_id
            df.loc[mask, c_status] = status
            df.loc[mask, [f"{stem}_{model_name}" for stem in stems.keys()]] = stems.values()

    logger.info("Source separation finished successfully.")
    return df, stems_list
