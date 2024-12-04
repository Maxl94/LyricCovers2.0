import base64
import json
import logging
import os
from typing import Optional

import functions_framework
import google.cloud.logging
import requests
import yt_dlp
from bs4 import BeautifulSoup
from google.cloud import pubsub_v1, storage

PROJECT_ID = "<google-cloud-project-id>"
BUECKT_NAME = "<google-cloud-project-id>-data"


class CustomCloudLogger(logging.Logger):
    def __init__(self, name: str, project: str, level=logging.INFO):
        super().__init__(name, level)
        client = google.cloud.logging.Client(project=project)
        self._logger = client.logger(name)

    def debug(self, msg, *args, **kwargs):
        self._log_client("DEBUG", msg, *args)

    def info(self, msg, *args, **kwargs):
        self._log_client("INFO", msg, *args)

    def warning(self, msg, *args, **kwargs):
        self._log_client("WARNING", msg, *args)

    def error(self, msg, *args, **kwargs):
        self._log_client("ERROR", msg, *args)

    def critical(self, msg, *args, **kwargs):
        self._log_client("CRITICAL", msg, *args)

    def exception(self, msg, *args, **kwargs):
        self._log_client("CRITICAL", msg, *args)

    def _log_client(self, level, msg, *args):
        msg = msg % args
        self._logger.log(msg, severity=level)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    logger = CustomCloudLogger(name or "main", PROJECT_ID)
    return logger


YDL_OPS = {
    "format": "bestaudio/best",
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }
    ],
}

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
    print(
        "Send message to topic (%s), ID: %s Content: %s Content encoded: %s",
        topic,
        future.result(),
        message,
        message_bytes,
    )


def decode_message_data(data: str):
    """Decode message from pubsub

    Args:
        message (str): Message

    Returns:
        str: Decoded message
    """
    print("Decoding message data %s", data)
    message = base64.b64decode(data).decode("utf-8")
    print("Decoded message data %s", message)
    return message


def copy_local_file_to_storage(local_file_path: str, destination_bucket_name: str):
    """Google Cloud Function entry point.

    Args:
         local_file_path (str): Path to local file.
    """
    file_name = os.path.basename(local_file_path)

    destination_file_name = os.path.join("genius/audio/", file_name)

    # Initialize the Cloud Storage client
    storage_client = storage.Client()

    # Get or create the destination bucket
    destination_bucket = storage_client.get_bucket(destination_bucket_name)

    # Upload the local file to Cloud Storage
    blob = destination_bucket.blob(destination_file_name)
    blob.upload_from_filename(local_file_path)


@functions_framework.cloud_event
def download_video(cloud_event):
    """Download video from youtube

    Args:
        cloud_event (dict): Cloud event
    """
    print("Downloading video from youtube")
    message = json.loads(decode_message_data(cloud_event.data["message"]["data"]))
    print("Decoded message %s", message)

    youtube_url = message["youtube_url"]
    song_id = message["song_id"]

    print("Downloaded video from youtube %s", youtube_url)

    YDL_OPS["outtmpl"] = f"{song_id}.%(ext)s"

    with yt_dlp.YoutubeDL(YDL_OPS) as ydl:
        error_code = ydl.download([youtube_url])

    if error_code != 0:
        print("Error downloading video from youtube %s", youtube_url)
        return

    copy_local_file_to_storage(f"{song_id}.wav", BUECKT_NAME)
    print("Finished downloading video from youtube %s", youtube_url)


def get_tags(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        tags = soup.find("div", class_="SongTags__Container-xixwg3-1").get_text(separator="\n").split("\n")
        return tags
    else:
        print("Error scraping tags from genius.com %s", url)
        return []


@functions_framework.cloud_event
def scrape_tags(cloud_event):
    """Scrape tags from genius.com

    Args:
        cloud_event (dict): Cloud event
    """
    print("Scraping tags from genius.com")
    message = json.loads(decode_message_data(cloud_event.data["message"]["data"]))
    print("Decoded message:", message)

    print("Scraping tags")

    song_id = message["song_id"]
    url = message["url"]

    tags = get_tags(url)
    print(f"Found tags for song ID {song_id}: {tags}")

    print("Send results to save")
    send_message(
        "tags-crawled",
        json.dumps(
            {
                "song_id": song_id,
                "tags": tags,
            }
        ),
    )

    print("Finished scraping tags from genius.com")
