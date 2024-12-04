import os

import torch
from zenml import get_step_context
from zenml.client import Client

from thesis_csi.logging import get_logger

logger = get_logger(__name__)

try:
    from google.auth import default

except ImportError:
    logger.warning("Google Cloud SDK not installed. Skipping Google Cloud SDK imports.")


def get_device() -> str:
    """Get compute device.

    Returns:
        str: Device
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "cpu"
    logger.info(f"Using device {device}")

    return device


def update_model_tags(tags: list[str]):
    """Update model tags.

    Args:
        tags (list[str]): Tags
    """
    client = Client()
    model = get_step_context().model
    client.update_model_version(model.model_id, model.version, add_tags=tags)


def print_gcp_account():
    """Print account information."""

    # Get current credentials and project
    credentials, project_id = default()

    # Print the service account email or user account
    service_account_email = (
        credentials.service_account_email
        if hasattr(credentials, "service_account_email")
        else credentials.signer_email
    )
    logger.info(f"Current account: {service_account_email}")
    logger.info(f"Current project: {project_id}")


def authenticate_with_service_account(key_path: str):
    """
    Authenticate with GCP using a service account JSON key file.

    Args:
        key_path (str): Path to the service account JSON key file

    Returns:
        google.oauth2.service_account.Credentials: The GCP credentials object

    Example:
        credentials = authenticate_with_service_account("/path/to/key.json")
        print(f"Authenticated as: {credentials.service_account_email}")
    """
    from google.oauth2 import service_account

    if not os.path.isfile(key_path):
        credentials = service_account.Credentials.from_service_account_file("key.json")
        return credentials

    else:
        try:
            credentials = service_account.Credentials.from_service_account_file(key_path)
            print(f"Successfully authenticated as: {credentials.service_account_email}")
            return credentials
        except Exception as e:
            print(f"Failed to authenticate with service account key at {key_path}")
            raise e
