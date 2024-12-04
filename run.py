import os

import click
import dotenv

# from pipelines.feature_engineering import feature_engineering
from pipelines.feature_engineering import feature_preparation
from pipelines.model_training import model_training
from thesis_csi.logging import get_logger

dotenv.load_dotenv()

logger = get_logger(__name__)


@click.command()
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--test",
    is_flag=True,
    default=False,
    help="Run the pipeline on the cloud.",
)
@click.option(
    "-s",
    "--source-separation",
    type=click.Choice(["spleeter", "demucs"], case_sensitive=False),
    help="Choose the source separation model.",
    default=None,
)
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(["feature_engineering", "model_training"], case_sensitive=False),
    help="Choose which pipeline to run.",
)
def main(pipeline: str, no_cache: bool = False, test: bool = False, source_separation: str = None):
    """Run the pipeline."""

    config_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "configs",
    )
    pipeline_args = {}

    config_file = os.path.join(config_folder, f"{pipeline}")

    if source_separation == "spleeter":
        # TODO: set transcription dataset id
        source_separation_config = {
            "c_status": "source_separation_status_spleeter",
            "c_vocals": "vocals_spleeter",
        }
        # transcription_id = "cad2fc8a-5f2f-4805-b28e-503ed3decd1c"
    elif source_separation == "demucs":
        source_separation_config = {
            "c_status": "source_separation_status_htdemucs",
            "c_vocals": "vocals_htdemucs",
        }
        # transcription_id = "b6d72537-494f-4a3e-a549-b8ce6219ef99"
    else:
        # transcription_id = "0c0f7c66-c806-4be9-85f3-fcdc7f747bfb"
        pass

    if no_cache:
        pipeline_args["enable_cache"] = False

    if test:
        logger.info("Running pipeline with production config file.")
        pipeline_args["config_path"] = os.path.join(config_folder, f"{config_file}_test.yaml")
    else:
        logger.info("Running pipeline with test config file")
        pipeline_args["config_path"] = os.path.join(config_folder, f"{config_file}.yaml")

    logger.info(f"Running pipeline with config file:\n{pipeline_args['config_path']}")

    if pipeline == "feature_engineering":
        logger.info("Running pipelines 'source_separation'")
        feature_preparation.with_options(**pipeline_args)()
    elif pipeline == "model_training":
        logger.info("Running pipelines 'model_training'")

        if source_separation is not None:
            logger.info(f"Running with source separation config: \n    {source_separation_config}")
            model_training.with_options(**pipeline_args)(
                source_separation_config=source_separation_config,
                # transcription_id=transcription_id,
            )
        else:
            model_training.with_options(**pipeline_args)(
                # transcription_id=transcription_id,
            )

    # Create a run
    logger.info("Pipeline run completed.")


if __name__ == "__main__":
    main()
