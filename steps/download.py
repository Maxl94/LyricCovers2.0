import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import dotenv
import pandas as pd
from google.cloud import pubsub_v1
from tqdm import tqdm
from zenml import log_artifact_metadata, step

from thesis_csi.analysis.releases import get_release_years
from thesis_csi.logging import get_logger
from thesis_csi.shared.pub_sub import decode_message_data, send_message
from thesis_csi.shared.utils import authenticate_with_service_account

dotenv.load_dotenv()
logger = get_logger(__name__)


@step()
def download_from_yt(df: pd.DataFrame, source: Optional["str"] = "") -> pd.DataFrame:
    """Download audio from YouTube.

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame
    """
    topic = "download-video-response"
    project_id = os.environ["PROJECT_ID"]
    retriever_topic_name = f"projects/{project_id}/topics/{topic}"

    credentials = authenticate_with_service_account(
        "/gcs/<google-cloud-project-id>-data/<google-cloud-project-id>-ce344196feb8.json"
    )

    publisher = pubsub_v1.PublisherClient()

    df["youtube_download_status"] = None
    df["youtube_download_gs_path"] = None

    # Update last message processing time
    last_message_time = datetime.now()
    received_messages = 0

    def callback(message):
        """Callback function for subscriber"""
        nonlocal last_message_time, received_messages
        decoded_message = decode_message_data(message.data)
        logger.info("Received message: {}".format(decoded_message))
        data = json.loads(decoded_message)

        df.loc[df["id"] == data["song_id"], "youtube_download_status"] = data["status"]
        df.loc[df["id"] == data["song_id"], "youtube_download_gs_path"] = data.get("song_path", None)
        message.ack()

        # Update last message processing time
        last_message_time = datetime.now()
        received_messages += 1

    with pubsub_v1.SubscriberClient(credentials=credentials) as subscriber:
        logger.info(f"Creating subscription for {retriever_topic_name}")
        subscription = subscriber.create_subscription(topic=retriever_topic_name)
        future = subscriber.subscribe(subscription.name, callback)

        logger.info("Listening for messages on {}..\n".format(subscription.name))
        try:
            for _, row in df[["id", "youtube_url"]].iterrows():
                send_message(
                    "youtube",
                    json.dumps({"song_id": row["id"], "youtube_url": row["youtube_url"], "source": source}),
                    publisher,
                    project_id,
                )

            logger.info(f"Sent {len(df)} messages to {topic}")

            while received_messages < len(df):
                # Check if no message was processed for more than one hour
                if (datetime.now() - last_message_time) > timedelta(minutes=10):
                    logger.info("No message was processed for more than one 10 minutes")
                    break
                logger.info(f"Received {received_messages}/{len(df)} messages")
                time.sleep(1)
        finally:
            future.cancel()

    log_artifact_metadata(
        metadata={
            "Number of rows": len(df),
            "Number of successful downloads": len(df[df["youtube_download_status"] == "success"]),
            "Number of failed downloads": len(df[df["youtube_download_status"] == "error"]),
        }
    )

    return df


@step()
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with failed YouTube downloads and remove covers without originals and vice
    versa.

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame
    """

    df = df[df["youtube_download_status"] == "success"]

    df_covers = df[df["is_cover"]]
    df_originals = df[~df["is_cover"]]

    # Remove covers with no original
    df_covers_complete = df_covers[df_covers["original_id"].isin(df_originals["id"])]
    logger.info(f"Removed {len(df_covers) - len(df_covers_complete)} covers with no original")

    # Remove originals with no covers
    df_originals_complete = df_originals[df_originals["id"].isin(df_covers_complete["original_id"])]
    logger.info(f"Removed {len(df_originals) - len(df_originals_complete)} originals with no covers")

    df_all = pd.concat([df_covers_complete, df_originals_complete])

    log_artifact_metadata(
        metadata={
            "Number of rows": len(df_all),
            "Number of cover songs": len(df_covers_complete),
            "Number of original songs": len(df_originals_complete),
        }
    )

    return df_all


@step
def get_release_year(df: pd.DataFrame) -> pd.DataFrame:
    """Get release year from the release date."""
    df["release_year"] = get_release_years(df)
    return df


@step(enable_step_logs=False)
def crawl_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Crawl tags from YouTube API."""

    topic = "tags-crawled"
    project_id = os.getenv("PROJECT_ID")
    topic_name = f"projects/{project_id}/topics/{topic}"

    credentials = authenticate_with_service_account(
        "/gcs/<google-cloud-project-id>-data/<google-cloud-project-id>-ce344196feb8.json"
    )

    last_message_time = datetime.now()
    received_messages = 0

    publisher = pubsub_v1.PublisherClient()

    def callback(message):
        nonlocal last_message_time, received_messages

        decoded_message = decode_message_data(message.data)
        print("Received message: {}".format(decoded_message))
        data = json.loads(decoded_message)

        df.loc[df["id"] == data["song_id"], "tags"] = "; ".join(data["tags"])
        message.ack()
        # Update last message processing time
        last_message_time = datetime.now()
        received_messages += 1

    with pubsub_v1.SubscriberClient(credentials=credentials) as subscriber:
        subscription = subscriber.create_subscription(topic=topic_name)
        future = subscriber.subscribe(subscription.name, callback)

        data = df[["id", "url"]]

        for _, row in tqdm(data.iterrows(), total=len(data)):
            send_message(
                "tags-urls",
                json.dumps({"song_id": row["id"], "url": row["url"]}),
                publisher,
                os.environ["PROJECT_ID"],
            )

        print("Listening for messages on {}..\n".format(subscription.name))

        try:
            while received_messages < len(df):
                # Check if no message was processed for more than one hour
                if (datetime.now() - last_message_time) > timedelta(minutes=10):
                    logger.info("No message was processed for more than one 10 minutes")
                    break
                logger.info(f"Received {received_messages}/{len(df)} messages")
                time.sleep(1)

        finally:
            future.cancel()

    return df
