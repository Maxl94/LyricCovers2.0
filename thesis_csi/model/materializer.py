import json
import os
from typing import Type

import torch
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from thesis_csi.model.model import LyricsModel


class LyricsModelMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = [LyricsModel]
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL

    def load(self, data_type: Type[LyricsModel]) -> LyricsModel:
        with fileio.open(os.path.join(self.uri, "model_meta.json"), "r") as f:
            obj = json.loads(f.read())
            model = LyricsModel(**obj["hparams"])

        with fileio.open(os.path.join(self.uri, "model.pt"), "rb") as f:
            model.load_state_dict(torch.load(f, map_location="cpu"))
            return model

    def save(self, data: LyricsModel) -> None:
        fileio.makedirs(self.uri)

        with fileio.open(os.path.join(self.uri, "model.pt"), "wb") as f:
            torch.save(data.state_dict(), f)

        with fileio.open(os.path.join(self.uri, "model_meta.json"), "w") as f:
            obj = {
                "hparams": dict(data.hparams),
            }
            f.write(json.dumps(obj))
