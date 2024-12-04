import pandas as pd
from zenml import step
from zenml.config.resource_settings import ResourceSettings
from zenml.materializers.pandas_materializer import PandasMaterializer
from zenml.materializers.structured_string_materializer import StructuredStringMaterializer
from zenml.types import CSVString

from thesis_csi.analysis.lyrics.similarity.calculation import (
    SimilarityCalculator,
    SimilarityMetrics,
)
from thesis_csi.materializers import DictPandas


@step(
    output_materializers=[DictPandas, PandasMaterializer, StructuredStringMaterializer],
    enable_cache=False,
    settings={
        "resources": ResourceSettings(cpu_count=4, memory="64GiB"),
    },
)
def lyric_similarity(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    CSVString,
    CSVString,
]:
    similarity_calculator = SimilarityCalculator(
        metrics=[
            SimilarityMetrics.LEVENSHTEIN,
            SimilarityMetrics.HAMMING,
            SimilarityMetrics.JARO,
            SimilarityMetrics.JARO_WINKLER,
            SimilarityMetrics.BLEU,
            SimilarityMetrics.WER,
            SimilarityMetrics.VOCABULARY,
        ]
    )

    similarities = similarity_calculator.compute_cover_original_metrics_v2(df)
    control_group = similarity_calculator.compute_control_group_metrics_v2(df)

    similarities_cover_original_csv = CSVString(similarities.describe().T.to_csv())
    control_group_csv = CSVString(pd.concat(control_group, axis=1).describe().T.to_csv())

    return (
        similarities,
        control_group,
        similarities_cover_original_csv,
        control_group_csv,
    )
