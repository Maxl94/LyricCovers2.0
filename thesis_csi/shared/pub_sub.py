from datetime import datetime

from google.cloud import pubsub_v1


def decode_message_data(data: str):
    """Decode message from pubsub

    Args:
        message (str): Message

    Returns:
        str: Decoded message
    """
    message = data.decode("utf-8")
    return message


def send_message(topic: str, message: str, publisher: pubsub_v1.PublisherClient, project_id: str):
    """Send message to pubsub topic

    Args:
        topic (str): Topic name
        message (str): Message
        publisher (pubsub_v1.PublisherClient): Publisher client
        project_id (str): Project ID
    """

    topic_path = publisher.topic_path(project_id, topic)
    message_bytes = message.encode("utf-8")
    publisher.publish(topic_path, data=message_bytes)


# Define a global variable to keep track of the last message processing time
last_message_time = datetime.now()
