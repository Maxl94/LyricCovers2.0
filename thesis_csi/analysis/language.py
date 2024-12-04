import pandas as pd


def top_10_languages_of_originals(df: pd.DataFrame, normalize: bool = False) -> pd.Series:
    """Get top 10 languages of original songs.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Top 10 languages of original songs
    """

    df_org_lang = df[~df["is_cover"]].value_counts("language", normalize=normalize)

    return df_org_lang.head(10)


def top_10_languages_of_covers(df: pd.DataFrame, normalize: bool = False) -> pd.Series:
    """Get top 10 languages of cover songs.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        pd.Series: Top 10 languages of cover songs
    """

    df_cover_lang = df[df["is_cover"]].value_counts("language", normalize=normalize)

    return df_cover_lang.head(10)


def top_10_languages_combined(
    df: pd.DataFrame, normalize: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Get top 10 languages of original and cover songs.

    Args:
        df (pd.DataFrame): DataFrame containing original and cover songs

    Returns:
        tuple[pd.Series, pd.Series]: Top 10 languages of original and cover songs
    """

    return df.value_counts("language", normalize=normalize).head(10)
