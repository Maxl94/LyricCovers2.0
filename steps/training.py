import os
from datetime import datetime
from typing import Literal, Optional

import lightning as L
import pandas as pd
import torch
from typing_extensions import Annotated
from zenml import get_step_context, log_artifact_metadata, step
from zenml.config.resource_settings import ResourceSettings
from zenml.integrations.gcp.flavors.vertex_orchestrator_flavor import VertexOrchestratorSettings

from thesis_csi.enum import Labels
from thesis_csi.logging import get_logger
from thesis_csi.model.dataset import EmbeddingDataModule
from thesis_csi.model.materializer import LyricsModelMaterializer
from thesis_csi.model.model import LyricsModel
from thesis_csi.shared.utils import update_model_tags

logger = get_logger(__name__)


@step(
    settings={
        "resources": ResourceSettings(cpu_count=4, gpu_count=1, memory="8GiB"),
        "orchestrator.vertex": VertexOrchestratorSettings(
            node_selector_constraint=("cloud.google.com/gke-accelerator", "NVIDIA_L4")
        ),
    },
    output_materializers=LyricsModelMaterializer,
)
def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    base_model: str,
    batch_size: int = 32,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    patience_early_stopping: int = 10,
    lr_reduce_patience: int = 5,
    min_delta_early_stopping: float = 0.0,
    seed: int = 42,
    lr_warmup: bool = False,
    max_length: int = 512,
    scale_batch_size: Optional[str] = None,
    trainable_layers: Optional[list[str]] = None,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 2.0,
    sampler_m: int = 4,
    loss_fn: Literal["arcface", "triplet"] = "triplet",
) -> Annotated[LyricsModel, "model"]:
    """Train model.

    Args:
        df_train (pd.DataFrame): Training DataFrame
        df_val (pd.DataFrame): Validation DataFrame
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        logger.info("Trying to import Torch XLA")
        os.environ["PJRT_DEVICE"] = "TPU"

        import torch_xla.core.xla_model as xm  # noqa: reportMissingImports

        logger.info(f"Training on TPU: {xm.xla_device()}")

    except ImportError:
        logger.warning("Torch XLA not available.")

    L.seed_everything(seed)

    update_model_tags([base_model])

    logger.info(f"Training with batch size: {batch_size}")
    logger.info(f"Training samples: {len(df_train)}\nValidation samples: {len(df_val)}")

    tensorboard_logger = L.pytorch.loggers.TensorBoardLogger(
        save_dir="/gcs/<google-cloud-project-id>-data/training" if torch.cuda.is_available() else "logs",
        version=get_step_context().model.version + "_" + datetime.now().strftime("%Y%m%d-%H%M"),
    )
    early_stop_callback = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience_early_stopping,
        mode="min",
        verbose=True,
        min_delta=min_delta_early_stopping,
    )
    model_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    callbacks = [early_stop_callback, model_checkpoint_callback]

    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        logger=tensorboard_logger,
        callbacks=callbacks,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
        enable_progress_bar=False if torch.cuda.is_available() else True,
    )

    model = LyricsModel(
        base_model_name=base_model,
        loss_fn=loss_fn,
        learning_rate=learning_rate,
        lr_reduce_patience=lr_reduce_patience,
        lr_reduce_threshold=min_delta_early_stopping,
        max_length=max_length,
    )

    if trainable_layers:
        logger.info(f"Freezing all layers except {len(trainable_layers)}")
        model.freeze_base_model(trainable_layers)
    else:
        logger.info("Freezing all layers")
        model.freeze_base_model()

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info("Setting float32 matmul precision to medium")
        torch.set_float32_matmul_precision("medium")

        logger.info("Cuda available. Compiling model for Triton")
        torch.compile(model, options={"triton.cudagraphs": True}, fullgraph=True)

    datamodule = EmbeddingDataModule(
        base_model=base_model,
        df_train=df_train,
        df_val=df_val,
        df_test=df_val,
        batch_size=batch_size,
        max_length=model.hparams.max_length,
        sampler_m=sampler_m,
    )
    datamodule.setup("train")
    datamodule.setup("val")

    if lr_warmup or scale_batch_size:
        tuner = L.pytorch.tuner.Tuner(trainer)

    if lr_warmup:
        logger.info("Warm-up learning rate")
        tuner.lr_find(model, datamodule)
        model.save_hyperparameters()

    if scale_batch_size:
        assert scale_batch_size in ["power", "binsearch"]
        logger.info("Scaling batch size")
        tuner.scale_batch_size(model, mode=scale_batch_size, datamodule=datamodule)

    logger.info("Training model")
    trainer.fit(model, datamodule)

    train_loss = float(trainer.callback_metrics["train_loss"])
    val_loss = float(trainer.callback_metrics["val_loss"])

    logger.info("Model trained successfully")
    best_model = LyricsModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    log_artifact_metadata(
        artifact_name="model",
        metadata={
            "training_loss": train_loss,
            "validation_loss": val_loss,
            "training_samples": len(df_train),
            "validation_samples": len(df_val),
            "training_products": df_train[Labels.ORIGINAL_SONG_ID.column].nunique(),
            "validation_products": df_val[Labels.ORIGINAL_SONG_ID.column].nunique(),
        },
    )

    return best_model
