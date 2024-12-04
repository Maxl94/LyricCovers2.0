import json
import os
from typing import Type

import pandas as pd
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer


class DictPandas(BaseMaterializer):
    ASSOCIATED_TYPES = [dict]
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def load(self, data_type: Type[dict]) -> dict:
        data = {}

        with fileio.open(os.path.join(self.uri, "keys.json"), "r") as f:
            keys = json.load(f)

        for key in keys:
            with fileio.open(os.path.join(self.uri, f"{key}.parquet"), "rb") as f:
                data[key] = pd.read_parquet(f)

        return data

    def save(self, data: dict) -> None:
        fileio.makedirs(self.uri)
        for key, value in data.items():
            with fileio.open(os.path.join(self.uri, f"{key}.parquet"), "wb") as f:
                value.to_parquet(f)

        with fileio.open(os.path.join(self.uri, "keys.json"), "w") as f:
            json.dump(list(data.keys()), f)
