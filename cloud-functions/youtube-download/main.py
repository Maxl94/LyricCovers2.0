import base64
import json
import logging
import os
import sys
import time

import functions_framework
import yt_dlp
from dotenv import load_dotenv
from google.cloud import pubsub_v1, storage

load_dotenv()


class MyLogger:
    def debug(self, msg):
        if msg.startswith("[debug] "):
            logger.debug(msg)
        else:
            self.info(msg)

    def info(self, msg):
        logger.info(msg)

    def warning(self, msg):
        logger.warning(msg)

    def error(self, msg):
        logger.error(msg)


PROJECT_ID = os.environ["PROJECT_ID"]
BUECKT_NAME = os.environ["BUECKT_NAME"]

RESPONSE_TOPIC = "download-video-response"
MAX_INVOCATIONS = 10


YDL_OPS = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }
    ],
    "logger": MyLogger(),
}


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

invocation_count = 0


# Setup pubsub
publisher = pubsub_v1.PublisherClient()


# General functions
def send_message(topic: str, message: str):
    """Send message to pubsub topic

    Args:
        topic (str): Topic name
        message (str): Message
    """

    topic_path = publisher.topic_path(PROJECT_ID, topic)
    message_bytes = message.encode("utf-8")
    future = publisher.publish(topic_path, data=message_bytes)
    logger.info(
        f"Send message to topic ({topic}), ID: {future.result()} Content: {message} Content encoded: {message_bytes}"
    )


def decode_message_data(data: str):
    """Decode message from pubsub

    Args:
        message (str): Message

    Returns:
        str: Decoded message
    """
    logger.info(f"Decoding message data {data}")
    message = base64.b64decode(data).decode("utf-8")
    logger.info(f"Decoded message data {message}")
    return message


def copy_local_file_to_storage(local_file_path: str, destination_bucket_name: str, source: str):
    """Google Cloud Function entry point.

    Args:
         local_file_path (str): Path to local file.
    """
    file_name = os.path.basename(local_file_path)
    destination_file_name = os.path.join(source, file_name)

    logger.info(f"Copying {local_file_path} to {destination_file_name}")

    # Initialize the Cloud Storage client
    storage_client = storage.Client()

    # Get or create the destination bucket
    destination_bucket = storage_client.get_bucket(destination_bucket_name)

    # Upload the local file to Cloud Storage
    blob = destination_bucket.blob(destination_file_name)
    blob.upload_from_filename(local_file_path)
    return f"gs://{destination_bucket_name}/{destination_file_name}"


def file_exists_in_bucket(bucket_name: str, file_name: str):
    """Check if file exists in bucket

    Args:
        bucket_name (str): Bucket name
        file_name (str): File name

    Returns:
        bool: True if file exists, False otherwise
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    return blob.exists()


@functions_framework.cloud_event
def download_video(cloud_event):
    """Download video from youtube

    Args:
        cloud_event (dict): Cloud event
    """
    logger.info("Downloading video from youtube")
    message = json.loads(decode_message_data(cloud_event.data["message"]["data"]))
    logger.info(f"Decoded message {message}")

    youtube_url = str(message["youtube_url"])
    song_id = message["song_id"]
    source = str(message.get("source", "genius/v2/audio/"))
    logger.info(f"Source: {source}")
    logger.info(f"URL type: {type(youtube_url)}")

    yt_download = False

    try:
        logger.info(f"Downloaded video from youtube {youtube_url}")

        YDL_OPS["outtmpl"] = f"{song_id}.%(ext)s"

        if source == "genius":
            print("Using genius source v2")
            source = "genius/v2/audio/"

        file_name = os.path.join(source, f"{song_id}.wav")

        print(f"Checking if file {file_name} exists in bucket {BUECKT_NAME}")
        if file_exists_in_bucket(BUECKT_NAME, file_name):
            print(f"File {file_name} already exists in bucket {BUECKT_NAME}, skipping download")
            file_path = f"gs://{BUECKT_NAME}/{file_name}"
        else:
            print(f"File {file_name} does not exist in bucket {BUECKT_NAME}, downloading")

            yt_download = True
            with yt_dlp.YoutubeDL(YDL_OPS) as ydl:
                error_code = ydl.download([youtube_url])

            if error_code != 0:
                logger.error(f"Error downloading video from youtube {youtube_url} {error_code}")
                send_message(RESPONSE_TOPIC, json.dumps({"song_id": song_id, "status": "error"}))

            else:
                logger.info(f"Copied file {song_id}.wav to bucket {BUECKT_NAME}")
                file_path = copy_local_file_to_storage(f"{song_id}.wav", BUECKT_NAME, source=source)

        logger.info(f"Video available for {youtube_url}")
        send_message(
            RESPONSE_TOPIC,
            json.dumps(
                {
                    "song_id": song_id,
                    "status": "success",
                    "song_path": file_path,
                }
            ),
        )

    except (yt_dlp.utils.ExtractorError, yt_dlp.utils.DownloadError):
        logger.error(f"Youtube download error for song {song_id}")
        send_message(RESPONSE_TOPIC, json.dumps({"song_id": song_id, "status": "error"}))
    except Exception:
        logger.error(f"An exception occurred for song {song_id}", exc_info=True, stack_info=True)
        send_message(RESPONSE_TOPIC, json.dumps({"song_id": song_id, "status": "error"}))

    if yt_download:
        logger.info("Sleeping for 30 seconds, to avoid rate limiting")
        time.sleep(60)

    if invocation_count >= MAX_INVOCATIONS:
        logger.error("Invocation count exceeded, exiting")
        sys.exit(0)
