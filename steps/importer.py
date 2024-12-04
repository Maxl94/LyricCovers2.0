import pandas as pd
from zenml import step


@step
def load_dataset(gcs_path: str) -> pd.DataFrame:
    """Load dataset from GCS.

    Args:
        gcs_path (str): GCS path

    Returns:
        pd.DataFrame: DataFrame
    """
    return pd.read_parquet(gcs_path)
