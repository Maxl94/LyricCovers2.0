import pandas as pd

from thesis_csi.analysis.lyrics.splits import split_by_language_match

# FILE: thesis_csi/analysis/lyrics/test_splits.py


def test_split_by_language_match_same_language():
    data = {
        "id": [1, 2, 3, 4],
        "original_id": [1, 1, 3, 3],
        "language": ["en", "en", "fr", "fr"],
        "is_cover": [False, True, False, True],
    }
    df = pd.DataFrame(data)
    same_language_df, different_language_df = split_by_language_match(df)

    assert len(same_language_df) == 2
    assert len(different_language_df) == 0


def test_split_by_language_match_different_language():
    data = {
        "id": [1, 2, 3, 4],
        "original_id": [1, 1, 3, 3],
        "language": ["en", "fr", "fr", "en"],
        "is_cover": [False, True, False, True],
    }
    df = pd.DataFrame(data)
    same_language_df, different_language_df = split_by_language_match(df)

    assert len(same_language_df) == 0
    assert len(different_language_df) == 2


def test_split_by_language_match_no_covers():
    data = {
        "id": [1, 2],
        "original_id": [1, 2],
        "language": ["en", "fr"],
        "is_cover": [False, False],
    }
    df = pd.DataFrame(data)
    same_language_df, different_language_df = split_by_language_match(df)

    assert len(same_language_df) == 0
    assert len(different_language_df) == 0


def test_split_by_language_match_no_originals():
    data = {
        "id": [1, 2],
        "original_id": [1, 2],
        "language": ["en", "fr"],
        "is_cover": [True, True],
    }
    df = pd.DataFrame(data)
    same_language_df, different_language_df = split_by_language_match(df)

    assert len(same_language_df) == 0
    assert len(different_language_df) == 2
