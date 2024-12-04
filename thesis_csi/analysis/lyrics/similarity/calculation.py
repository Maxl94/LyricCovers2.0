from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Set, Union

import evaluate
import Levenshtein as lev
import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis_csi.logging import get_logger

logger = get_logger(__name__)


class SimilarityMetrics(Enum):
    """Supported metrics for lyrics similarity comparison."""

    BLEU = "bleu"
    ROUGE = "rouge"  # Combines ROUGE1, ROUGE2, and ROUGE_L
    WER = "wer"

    VOCABULARY = "vocabulary"

    LEVENSHTEIN = "levenshtein"
    HAMMING = "hamming"
    JARO = "jaro"
    JARO_WINKLER = "jaro_winkler"

    # FIXME: This is very bad!!!
    @classmethod
    def get_library_metrics(cls, metrics: Set["SimilarityMetrics"]) -> Set[str]:
        """Get the set of required library names for the given metrics."""
        library_metrics = set()
        for metric in set(metrics):
            if metric not in cls.get_self_defined_metrics():
                library_metrics.add(metric.value)
        return library_metrics

    # FIXME: This is very bad!!!
    @classmethod
    def get_self_defined_metrics(cls) -> Set["SimilarityMetrics"]:
        """Get the set of self-defined metrics."""
        return {cls.VOCABULARY, cls.LEVENSHTEIN, cls.HAMMING, cls.JARO, cls.JARO_WINKLER}


@dataclass
class ComparisonMapping:
    """Data structure to hold original song and its covers."""

    base_id: str
    comparison_ids: List[str]
    base_lyrics: str
    comparison_lyrics: List[str]
    base_lang: str
    comparison_lang: List[str]


class SimilarityCalculator:
    """Calculate similarity metrics for cover songs using selected metrics."""

    def __init__(self, metrics: Union[Set[SimilarityMetrics], None] = None, num_process: int = 1):
        if metrics is None:
            metrics = set(list(SimilarityMetrics))

        self.selected_metrics = metrics
        self.metrics_dict = {
            lib_name: evaluate.load(lib_name, num_process=num_process)
            for lib_name in SimilarityMetrics.get_library_metrics(metrics)
        }

        try:
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()

            self.openai_client = OpenAI()
        except ImportError:
            logger.warning("OpenAI library not found. Translation will not be available.")

    def compute_cover_original_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate similarity metrics for all cover songs and their originals in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns "id", "original_id", "lyrics", "is_cover"

        Returns:
            pd.DataFrame: DataFrame with similarity metrics for each cover song
        """

        return self._calculate_similarity_metrics(
            self._create_cover_original_mappings(df), self.selected_metrics
        )

    def compute_cover_original_metrics_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate similarity metrics for all cover songs and their originals in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns "id", "original_id", "lyrics", "is_cover"

        Returns:
            pd.DataFrame: DataFrame with similarity metrics for each cover song
        """

        return self._calculate_similarity_metrics_v2(
            self._create_cover_original_mappings(df), self.selected_metrics
        )

    def compute_cover_original_metrics_translation(
        self, df: pd.DataFrame, sample_size: int = None
    ) -> pd.DataFrame:
        """Calculate similarity metrics for all cover songs and their originals in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with columns "id", "original_id", "lyrics", "is_cover"

        Returns:
            pd.DataFrame: DataFrame with similarity metrics for each cover song
        """

        return self._calculate_similarity_metrics_v2(
            self._create_cover_original_mappings(df),
            self.selected_metrics,
            sample_size=sample_size,
            use_translation=True,
        )

    def compute_control_group_metrics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate similarity metrics for control groups."""
        control_groups = {}

        # Artist-based control group
        # Covers of the same original artist
        original_artist_mappings = self._control_group_artist_based(df, by="original_artist")
        control_groups["original_artist"] = self._calculate_similarity_metrics(
            original_artist_mappings, self.selected_metrics
        )

        # Covers by the same cover artist
        cover_artist_mappings = self._control_group_artist_based(df, by="cover_artist")
        control_groups["cover_artist"] = self._calculate_similarity_metrics(
            cover_artist_mappings, self.selected_metrics
        )

        # Year-based control group
        # Covers of the same original release year
        original_year_mappings = self._control_group_year_based(df, by="original_year")
        control_groups["original_year"] = self._calculate_similarity_metrics(
            original_year_mappings, self.selected_metrics
        )

        # Covers of the same cover release year
        cover_year_mappings = self._control_group_year_based(df, by="cover_year")
        control_groups["cover_year"] = self._calculate_similarity_metrics(
            cover_year_mappings, self.selected_metrics
        )

        return control_groups

    def compute_control_group_metrics_v2(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate similarity metrics for control groups."""
        control_groups = {}

        # Artist-based control group
        # Covers of the same original artist
        original_artist_mappings = self._control_group_artist_based(df, by="original_artist")
        control_groups["original_artist"] = self._calculate_similarity_metrics_v2(
            original_artist_mappings, self.selected_metrics
        )

        # Covers by the same cover artist
        cover_artist_mappings = self._control_group_artist_based(df, by="cover_artist")
        control_groups["cover_artist"] = self._calculate_similarity_metrics_v2(
            cover_artist_mappings, self.selected_metrics
        )

        # Year-based control group
        # Covers of the same original release year
        original_year_mappings = self._control_group_year_based(df, by="original_year")
        control_groups["original_year"] = self._calculate_similarity_metrics_v2(
            original_year_mappings, self.selected_metrics
        )

        # Covers of the same cover release year
        cover_year_mappings = self._control_group_year_based(df, by="cover_year")
        control_groups["cover_year"] = self._calculate_similarity_metrics_v2(
            cover_year_mappings, self.selected_metrics
        )

        return control_groups

    def _create_cover_original_mappings(self, df: pd.DataFrame) -> List[ComparisonMapping]:
        """Convert DataFrame to list of OriginalCoverMapping objects."""
        originals = df[~df["is_cover"]]
        mappings = []

        for _, original in tqdm(
            originals.iterrows(), desc="Creating mappings", total=len(originals)
        ):
            covers = df[(df["is_cover"]) & (df["original_id"] == original["id"])]

            mapping = ComparisonMapping(
                base_id=original["id"],
                comparison_ids=covers["id"].tolist(),
                base_lyrics=original["lyrics"],
                comparison_lyrics=covers["lyrics"].tolist(),
                base_lang=original["language"],
                comparison_lang=covers["language"].tolist(),
            )
            mappings.append(mapping)

        return mappings

    def _control_group_artist_based(
        self, df: pd.DataFrame, by: Literal["original_artist", "cover_artist"]
    ) -> List[ComparisonMapping]:
        """"""
        covers = df[df["is_cover"]]
        originals = df[~df["is_cover"]]
        mappings = []

        if by == "original_artist":
            for _, row in tqdm(
                covers[["id", "original_id", "lyrics", "language"]].iterrows(),
                total=len(covers),
                desc="Creating mappings by original artist",
            ):
                song_id, original_id, lyrics, language = row
                original_artist = originals.loc[originals["id"] == original_id, "artist"].values

                if len(original_artist) > 0:
                    original_artist = original_artist[0]

                    other_songs_by_original_artist = originals[
                        (originals["artist"] == original_artist) & (originals["id"] != original_id)
                    ]

                    if len(other_songs_by_original_artist) > 0:
                        mappings.append(
                            ComparisonMapping(
                                base_id=song_id,
                                base_lyrics=lyrics,
                                comparison_ids=other_songs_by_original_artist["id"].tolist(),
                                comparison_lyrics=other_songs_by_original_artist[
                                    "lyrics"
                                ].tolist(),
                                base_lang=language,
                                comparison_lang=other_songs_by_original_artist[
                                    "language"
                                ].tolist(),
                            )
                        )

        if by == "cover_artist":
            for _, row in tqdm(
                covers[["id", "lyrics", "artist", "language"]].iterrows(),
                total=len(covers),
                desc="Creating mappings by cover artist",
            ):
                song_id, lyrics, cover_artist, language = row

                other_songs_by_cover_artist = covers[
                    (covers["artist"] == cover_artist) & (covers["id"] != song_id)
                ]

                if len(other_songs_by_cover_artist) > 0:
                    mappings.append(
                        ComparisonMapping(
                            base_id=song_id,
                            base_lyrics=lyrics,
                            comparison_ids=other_songs_by_cover_artist["id"].tolist(),
                            comparison_lyrics=other_songs_by_cover_artist["lyrics"].tolist(),
                            base_lang=language,
                            comparison_lang=other_songs_by_cover_artist["language"].tolist(),
                        )
                    )

        return mappings

    def _control_group_year_based(
        self, df: pd.DataFrame, by: Literal["cover_year", "original_year"], max_songs: int = 100
    ) -> List[ComparisonMapping]:
        covers = df[df["is_cover"]]
        originals = df[~df["is_cover"]]
        mappings = []

        for _, row in tqdm(
            covers[["id", "lyrics", "original_id", "language"]].iterrows(),
            total=len(covers),
            desc="Creating mappings by release year",
        ):
            song_id, lyrics, original_id, language = row

            if by == "cover_year":
                original = df[df["id"] == original_id]

                if len(original) > 0:
                    original_release_year = original["release_year"].values[0]

                    other_cover_songs_same_year = covers[
                        (covers["release_year"] == original_release_year)
                        & (covers["id"] != song_id)
                    ]

                    if len(other_cover_songs_same_year) > 100:
                        other_cover_songs_same_year = other_cover_songs_same_year.sample(
                            max_songs, random_state=42
                        )

                    if len(other_cover_songs_same_year) > 0:
                        mappings.append(
                            ComparisonMapping(
                                base_id=song_id,
                                base_lyrics=lyrics,
                                comparison_ids=other_cover_songs_same_year["id"].tolist(),
                                comparison_lyrics=other_cover_songs_same_year["lyrics"].tolist(),
                                base_lang=language,
                                comparison_lang=other_cover_songs_same_year["language"].tolist(),
                            )
                        )
            elif by == "original_year":
                original = df[df["id"] == original_id]

                if len(original) > 0:
                    original_release_year = original["release_year"].values[0]

                    other_originals_some_year = originals[
                        (originals["release_year"] == original_release_year)
                        & (originals["id"] != original_id)
                    ]

                    if len(other_originals_some_year) > 100:
                        other_originals_some_year = other_originals_some_year.sample(
                            max_songs, random_state=42
                        )

                    if len(other_originals_some_year) > 0:
                        mappings.append(
                            ComparisonMapping(
                                base_id=song_id,
                                base_lyrics=lyrics,
                                comparison_ids=other_originals_some_year["id"].tolist(),
                                comparison_lyrics=other_originals_some_year["lyrics"].tolist(),
                                base_lang=language,
                                comparison_lang=other_originals_some_year["language"].tolist(),
                            )
                        )
        return mappings

    def _compute_metrics_for_pair(
        self,
        cover_text: str,
        original_text: str,
    ) -> Dict[str, float]:
        """Calculate selected similarity metrics between a cover and original lyrics pair."""
        results = {}

        if SimilarityMetrics.BLEU in self.selected_metrics:
            bleu_result = self.metrics_dict["bleu"].compute(
                predictions=[cover_text], references=[[original_text]]
            )
            results["bleu"] = bleu_result["bleu"]

        if SimilarityMetrics.ROUGE in self.selected_metrics:
            rouge_result = self.metrics_dict["rouge"].compute(
                predictions=[cover_text], references=[original_text]
            )
            # Include all ROUGE metrics when ROUGE is selected
            results.update(
                {
                    "rouge1": rouge_result["rouge1"],
                    "rouge2": rouge_result["rouge2"],
                    "rougeL": rouge_result["rougeL"],
                }
            )

        if SimilarityMetrics.WER in self.selected_metrics:
            wer_result = self.metrics_dict["wer"].compute(
                predictions=[cover_text], references=[original_text]
            )
            results["wer"] = wer_result

        if SimilarityMetrics.VOCABULARY in self.selected_metrics:
            words_original = set(original_text.split())
            words_cover = set(cover_text.split())

            words_of_cover_in_original = len(words_cover.intersection(words_original))
            words_in_original = len(words_original)

            results[SimilarityMetrics.VOCABULARY.value] = (
                words_of_cover_in_original / words_in_original
            )

        if SimilarityMetrics.LEVENSHTEIN in self.selected_metrics:
            results[SimilarityMetrics.LEVENSHTEIN.value] = lev.distance(original_text, cover_text)

        if SimilarityMetrics.HAMMING in self.selected_metrics:
            results[SimilarityMetrics.HAMMING.value] = lev.hamming(original_text, cover_text)

        if SimilarityMetrics.JARO in self.selected_metrics:
            results[SimilarityMetrics.JARO.value] = lev.jaro(original_text, cover_text)

        if SimilarityMetrics.JARO_WINKLER in self.selected_metrics:
            results[SimilarityMetrics.JARO_WINKLER.value] = lev.jaro_winkler(
                original_text, cover_text
            )
        return results

    def _calculate_similarity_metrics_v2(
        self,
        mappings: List[ComparisonMapping],
        selected_metrics: Set[SimilarityMetrics],
        use_translation: bool = False,
        sample_size: int = None,
    ) -> pd.DataFrame:
        logger.info("Creating DataFrame for comparison")
        records = []
        for m in mappings:
            for comp_id, comp_lyrics, comp_lang in zip(
                m.comparison_ids, m.comparison_lyrics, m.comparison_lang
            ):
                records.append(
                    {
                        "base_id": m.base_id,
                        "base_lyrics": m.base_lyrics,
                        "comparison_id": comp_id,
                        "comparison_lyrics": comp_lyrics,
                        "base_lang": m.base_lang,
                        "comparison_lang": comp_lang,
                    }
                )

        df = pd.DataFrame(
            records,
            columns=[
                "base_id",
                "base_lyrics",
                "comparison_id",
                "comparison_lyrics",
                "base_lang",
                "comparison_lang",
            ],
        )
        logger.warning(f"Drop {len(df) - len(df.dropna())} rows with NaN values")
        df = df.dropna()

        if use_translation:
            logger.info("Using only pairs with different languages")
            df = df[df["base_lang"] != df["comparison_lang"]]
            if sample_size and sample_size < len(df):
                logger.info(f"Sampling {sample_size} pairs")
                df = df.sample(sample_size, random_state=42)

            df["comparison_lyrics"] = df.apply(
                lambda x: self._translate(x["comparison_lyrics"], x["base_lang"]), axis=1
            )
            df.to_parquet("translated_lyrics.parquet")

        else:
            drop = len(df) - len(df[df["base_lang"] == df["comparison_lang"]])
            logger.warning(f"Dropping {drop} paris with out the same language ")
            df = df[df["base_lang"] == df["comparison_lang"]]

        logger.info(f"Preprocessing {len(df)} lyric pairs")
        df["base_lyrics"] = df["base_lyrics"].apply(self._preprocess_lyrics)
        df["comparison_lyrics"] = df["comparison_lyrics"].apply(self._preprocess_lyrics)

        if len(df) == 0:
            logger.warning("No lyrics to compare")
            return pd.DataFrame()

        logger.info("Calculating metrics")
        results = {}

        if SimilarityMetrics.BLEU in selected_metrics:
            logger.info("Calculating BLEU")
            bleu_results = self.metrics_dict["bleu"].compute(
                predictions=df["comparison_lyrics"].tolist(),
                references=df["base_lyrics"].tolist(),
            )
            results["bleu"] = bleu_results["bleu"]

        if SimilarityMetrics.ROUGE in selected_metrics:
            logger.info("Calculating ROUGE")
            rouge_results = self.metrics_dict["rouge"].compute(
                predictions=df["comparison_lyrics"].tolist(), references=df["base_lyrics"].tolist()
            )
            results["rouge1"] = rouge_results["rouge1"]
            results["rouge2"] = rouge_results["rouge2"]
            results["rougeL"] = rouge_results["rougeL"]

        if SimilarityMetrics.WER in selected_metrics:
            logger.info("Calculating WER")
            wer_results = self.metrics_dict["wer"].compute(
                predictions=df["comparison_lyrics"].tolist(), references=df["base_lyrics"].tolist()
            )
            results["wer"] = wer_results

        if SimilarityMetrics.VOCABULARY in selected_metrics:
            logger.info("Calculating vocabulary")
            df["base_words"] = df["base_lyrics"].apply(lambda x: set(x.split()))
            df["comparison_words"] = df["comparison_lyrics"].apply(lambda x: set(x.split()))
            df["words_of_cover_in_original"] = df.apply(
                lambda x: len(x["base_words"].intersection(x["comparison_words"])), axis=1
            )
            df["words_in_original"] = df["base_words"].apply(len)
            results["vocabulary"] = df["words_of_cover_in_original"] / df["words_in_original"]

        # Check if any distance metrics are selected
        distance_metrics = {
            SimilarityMetrics.LEVENSHTEIN: lev.distance,
            SimilarityMetrics.HAMMING: lev.hamming,
            SimilarityMetrics.JARO: lev.jaro,
            SimilarityMetrics.JARO_WINKLER: lev.jaro_winkler,
        }

        selected_distance_metrics = {
            metric: distance_func
            for metric, distance_func in distance_metrics.items()
            if metric in self.selected_metrics
        }

        if selected_distance_metrics:
            logger.info("Calculating Levenshtein, Hamming, Jaro, Jaro-Winkler")

            # Calculate all metrics at once using apply
            for metric, func in selected_distance_metrics.items():
                results[metric.value] = df.apply(
                    lambda row, func=func: func(row.base_lyrics, row.comparison_lyrics), axis=1
                )

        return pd.DataFrame(results)

    def _calculate_similarity_metrics(
        self, mappings: List[ComparisonMapping], selected_metrics: Set[SimilarityMetrics]
    ) -> pd.DataFrame:
        """Calculate selected similarity metrics for all cover songs using the mappings."""
        # Load only required metric libraries

        results = []
        total_items = sum(len(mapping.comparison_ids) for mapping in mappings)

        with tqdm(desc="Processing cover songs", total=total_items) as pbar:
            for mapping in mappings:
                original_text = self._preprocess_lyrics(mapping.base_lyrics)

                org_is_instrumental = "This song is an instrumental".lower() in original_text

                for comparison_id, comparison_lyrics in zip(
                    mapping.comparison_ids, mapping.comparison_lyrics
                ):
                    cover_text = self._preprocess_lyrics(comparison_lyrics)
                    cov_is_instrumental = "This song is an instrumental".lower() in cover_text

                    result = {"comparison_id": comparison_id, "base_id": mapping.base_id}

                    # Skip songs if both are instrumental
                    if (
                        cover_text
                        and original_text
                        and not (org_is_instrumental and cov_is_instrumental)
                    ):
                        result.update(self._compute_metrics_for_pair(cover_text, original_text))
                    else:
                        # Add NaN values for all selected metrics
                        nan_metrics = {metric.value: np.nan for metric in self.selected_metrics}
                        if SimilarityMetrics.ROUGE in selected_metrics:
                            del nan_metrics["rouge"]
                            nan_metrics.update(
                                {"rouge1": np.nan, "rouge2": np.nan, "rougeL": np.nan}
                            )
                        result.update(nan_metrics)

                    results.append(result)
                    pbar.update(1)

        return pd.DataFrame(results)

    def _preprocess_lyrics(self, text: str) -> str:
        """Clean and normalize lyrics text."""
        if pd.isna(text):
            return np.nan
        text = str(text).lower().strip()
        text = text.replace("\n", " ")
        text = "".join([" " if char.isdigit() else char for char in text])
        return " ".join(text.split()).strip()

    def _translate(self, text: str, lang: str) -> str:
        """Translate text to English."""
        try:
            chat_completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"Translate the following text from {lang} to English."
                        " Return only the translated text.",
                    },
                    {"role": "user", "content": text},
                ],
            )
            translation = chat_completion.choices[0].message.content
            return translation
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            return np.nan
