from typing import Literal, Union

import lightning as pl
import torch
from pytorch_metric_learning.losses import ArcFaceLoss
from transformers import AutoModel

from thesis_csi.logging import get_logger
from thesis_csi.model.loss import online_mine_hard

logger = get_logger(__name__)


class LyricsModel(pl.LightningModule):
    def __init__(
        self,
        base_model_name: str,
        loss_fn: Literal["arcface", "triplet"] = "triplet",
        learning_rate: float = 1e-3,
        lr_reduce_patience: int = 5,
        lr_reduce_threshold: float = 1e-4,
        max_length: int = 512,
        **kwargs: Union[int, str],
    ):
        super().__init__()
        # Save hyperparameters
        self.hparams.learning_rate = learning_rate
        self.hparams.lr_reduce_patience = lr_reduce_patience
        self.hparams.lr_reduce_threshold = lr_reduce_threshold
        self.hparams.base_model_name = base_model_name
        self.hparams.loss_fn = loss_fn
        self.hparams.max_length = max_length

        # Load base model
        self.base_model = AutoModel.from_pretrained(
            self.hparams.base_model_name, trust_remote_code=True
        )
        self.hparams.embedding_size = self.base_model.pooler.dense.out_features

        # Embedding visualization
        self.val_embeddings = []
        self.val_labels = []

        if self.hparams.loss_fn == "arcface":
            self.loss_fn = ArcFaceLoss(
                num_classes=kwargs["num_classes"],
                embedding_size=self.hparams.embedding_size,
            )

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.hparams.lr_reduce_patience,
                    threshold=self.hparams.lr_reduce_threshold,
                ),
                "monitor": "val_loss",
            },
        }

    def freeze_base_model(self, trainable_layers: list[str] = None):
        """Freeze the base model layers, expect for the layers in trainable_layers.

        Args:
            trainable_layers (list[str], optional): List of layers to unfreeze. Defaults to None.

        Returns:
            None
        """

        def param_name_starts_with(param_name, layers):
            return any([param_name.startswith(layer) for layer in layers])

        self.base_model.requires_grad_(False)

        if trainable_layers:
            for name, param in self.base_model.named_parameters():
                if param_name_starts_with(name, trainable_layers):
                    logger.info(f"Unfreezing layer: {name}")
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, tokens):
        embeddings = self.mean_pooling(self.base_model(**tokens), tokens["attention_mask"])
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def shared_step(self, batch, stage: str):
        tokens, labels = batch
        embeddings = self(tokens)

        # miner_output = self.miner_fn(embeddings, labels)
        if self.hparams.loss_fn == "triplet":
            loss = online_mine_hard(
                labels=labels, embeddings=embeddings, margin=0.05, device=self.device
            )
        elif self.hparams.loss_fn == "arcface":
            loss = self.loss_fn(embeddings, labels)

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        self.log("grad_norm", grad_norm)

        return embeddings, labels, loss

    def training_step(self, batch, batch_idx):
        embeddings, labels, loss = self.shared_step(batch, "train")
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, labels, loss = self.shared_step(batch, "val")
        if batch_idx <= 5:
            self.val_embeddings.append(embeddings)
            self.val_labels.append(labels)
        return loss

    def test_step(self, batch, batch_idx):
        embeddings, labels, loss = self.shared_step(batch, "test")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        tokens = batch
        embeddings = self(tokens)
        return embeddings

    def on_validation_epoch_end(self) -> None:
        embeddings = torch.vstack(self.val_embeddings)
        labels = torch.hstack(self.val_labels)

        self.log_embeddings(embeddings, labels, prefix="val")
        self.val_embeddings.clear()
        self.val_labels.clear()

    # Logging methods
    def log_embeddings(self, embeddings, labels, prefix: str = ""):
        """Log embeddings to tensorboard."""
        self.logger.experiment.add_embedding(
            embeddings,
            metadata=labels.tolist(),
            global_step=f"{self.current_epoch:0>4}_{prefix}",
        )
