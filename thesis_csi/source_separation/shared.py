from google.cloud import storage


def vocals_exist(song_id: str, source: str, model: str, bucket_name: str) -> bool:
    """Check if already a sperated vocals track existing int GCS.

    Args:
        song_id (str): Song ID
        source (str): Source
        model (str): Model

    Returns:
        bool: True if vocals exist
    """

    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(f"{source}/vocals/{song_id}.wav")
    return blob.exists()
