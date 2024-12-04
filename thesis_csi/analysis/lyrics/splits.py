import pandas as pd

from thesis_csi.logging import get_logger

logger = get_logger(__name__)


def split_by_language_match(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Split the dataset into two dataframes based on whether covers are in the same
    language as their originals. Uses self-join to get original song language.

    Args:
        df (pandas.DataFrame): DataFrame with columns "id", "original_id", "language", "is_cover"

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]: DataFrames with covers in the same and
          different language along with statistics.
    """
    # Get only cover songs
    covers_df = df[df["is_cover"]].copy()

    # Join with original songs to get original language
    # First, create a smaller DataFrame of original songs with just needed columns
    originals_df = df[~df["is_cover"]][["id", "language"]].copy()
    originals_df.columns = ["original_id", "original_language"]  # rename columns for merge

    # Merge covers with originals
    covers_with_original_lang = covers_df.merge(originals_df, on="original_id", how="left")

    # Create masks for language matching
    same_language_mask = (
        covers_with_original_lang["language"] == covers_with_original_lang["original_language"]
    )

    # Split into two dataframes
    df_same_language = covers_with_original_lang[same_language_mask].copy()
    df_different_language = covers_with_original_lang[~same_language_mask].copy()

    # Print statistics
    total = len(covers_with_original_lang)
    same = len(df_same_language)
    diff = len(df_different_language)

    stats = {
        "total": total,
        "same": same,
        "diff": diff,
        "nan": total - same - diff,
    }

    if total > 0:
        logger.info(f"Total covers: {total}")
        logger.info(f"Same language: {same} ({same/total*100:.1f}%)")
        logger.info(f"Different language: {diff} ({diff/total*100:.1f}%)")
        logger.info(f"NaN languages: {total - same - diff}")

    return df_same_language, df_different_language, stats
