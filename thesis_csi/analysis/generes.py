import os
from collections import Counter
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).parent
DEFAULT_GENRE_LIST = os.path.join(MODULE_DIR, "./assets/genres.list")


def get_unique_tags(df, tags_column="tags", return_with_count: bool = True):
    """Extract all unique tags from a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame containing tags
        tags_column (str): Name of column containing semicolon-separated tags
        sort_by_frequency (bool): If True, returns (tag, count) tuples sorted by frequency

    Returns:
        If sort_by_frequency=False: list of unique tags sorted alphabetically
        If sort_by_frequency=True: list of (tag, count) tuples sorted by frequency
    """
    all_tags = (
        ";".join(df[df[tags_column].notna()][tags_column].tolist()).replace("-", ";").split(";")
    )
    clean_tags = [tag.strip().lower() for tag in all_tags if tag.strip()]

    tag_counts = Counter(clean_tags)

    return (
        pd.DataFrame(tag_counts.items(), columns=["tag", "count"]).sort_values(
            by="count", ascending=False
        )
        if return_with_count
        else tag_counts.keys()
    )


def get_major_genre(tags, genre_list_path=None):
    """Get the major genre from tags based on priority list.

    Args:
        tags (str): Semicolon-separated string of tags
        genre_list_path (Path or str, optional): Path to genres priority list file.
            If None, uses default genres.list in the same directory as this module.

    Returns:
        str: Major genre or 'mixture' if multiple genres
    """
    if genre_list_path is None:
        genre_list_path = DEFAULT_GENRE_LIST

    with open(genre_list_path) as f:
        keep_list = f.read().splitlines()

    if not tags or pd.isna(tags):
        return "other"

    # Filter and clean tags
    tag_list = [
        tag.strip().lower()
        for tag in tags.replace("-", ";").split(";")
        if tag.lower().strip() in keep_list
    ]

    if not tag_list:
        return "other"
    elif len(tag_list) == 1:
        return tag_list[0]
    else:
        return "mixture"


def get_all_genres(tags, genre_list_path=None):
    """Get all genres from tags based on priority list.

    Args:
        tags (str): Semicolon-separated string of tags
        genre_list_path (Path or str, optional): Path to genres priority list file.
            If None, uses default genres.list in the same directory as this module.

    Returns:
        list: Sorted list of genres
    """
    if genre_list_path is None:
        genre_list_path = DEFAULT_GENRE_LIST

    with open(genre_list_path) as f:
        keep_list = f.read().splitlines()

    if not tags or pd.isna(tags):
        return ["other"]

    # Filter and clean tags
    tag_list = list(
        set(
            tag.strip().lower()
            for tag in tags.replace("-", ";").split(";")
            if tag.lower().strip() in keep_list
        )
    )

    if not tag_list:
        return ["other"]

    # Sort based on priority in keep_list
    priority = {tag: i for i, tag in enumerate(keep_list)}
    return sorted(tag_list, key=lambda x: priority.get(x.lower(), 0))
