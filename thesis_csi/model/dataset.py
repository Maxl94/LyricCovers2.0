from typing import Optional, Tuple

import lightning as L
import pandas as pd
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from thesis_csi.enum import Features, Labels


class PandasDataset:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_column: str = Features.SONG_TEXT.column,
        label_column: str = None,
    ):
        """Dataset for pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame.
            label_column (str): Label column for sorting and getting labels.
            cluster_column (str): Cluster column for sorting.
        """

        self.df = df
        self.label_column = label_column
        self.features_column = feature_column
        if self.label_column:
            self.df = df.sort_values(by=[label_column])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Get item by index.

        Args:
            idx (int): Index.

        Returns:
            Tuple[pd.Series, Optional[pd.Series]]: Features and labels. Labels only if label column
              is set.
        """
        if self.label_column is None:
            return (self.df.iloc[idx][self.features_column],)
        return self.df.iloc[idx][self.features_column], self.df.iloc[idx][self.label_column]


class EmbeddingDataModule(L.LightningDataModule):
    def __init__(
        self,
        base_model: str,
        df_train: pd.DataFrame = None,
        df_val: pd.DataFrame = None,
        df_test: pd.DataFrame = None,
        df_predict: pd.DataFrame = None,
        batch_size: int = 32,
        max_length: int = 1024,
        sampler_m: int = 4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_predict = df_predict

        self.batch_size = batch_size
        self.max_length = max_length
        self.sampler_m = sampler_m

    def setup(self, stage: str):
        if stage == "train":
            self._train_dataset = PandasDataset(self.df_train, label_column=Labels.LABEL.column)
        elif stage == "val":
            self._val_dataset = PandasDataset(self.df_val, label_column=Labels.LABEL.column)
        elif stage == "test":
            self._test_dataset = PandasDataset(self.df_test, label_column=Labels.LABEL.column)
        elif stage == "predict":
            self._predict_dataset = PandasDataset(self.df_predict)

    def collate_fn(self, batch):
        batch = [row[0] for row in batch]
        batch = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        return batch

    def collate_features_only(self, batch):
        return self.collate_fn(batch)

    def collate_with_label(self, batch):
        label = torch.tensor([row[1] for row in batch], dtype=torch.long)
        batch = self.collate_fn(batch)
        return batch, label

    def train_dataloader(self):
        length_before_new_iter = len(self._train_dataset) if len(self._train_dataset) >= self.batch_size else 100000
        sampler = MPerClassSampler(
            self._train_dataset.df[Labels.ORIGINAL_SONG_ID.column],
            m=self.sampler_m,
            batch_size=self.batch_size,
            length_before_new_iter=length_before_new_iter,
        )
        loader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
            sampler=sampler,
        )

        loader.collate_fn = self.collate_with_label
        return loader

    def val_dataloader(self):
        length_before_new_iter = len(self._val_dataset) if len(self._val_dataset) >= self.batch_size else 100000
        sampler = MPerClassSampler(
            self._val_dataset.df[Labels.ORIGINAL_SONG_ID.column],
            m=self.sampler_m,
            batch_size=self.batch_size,
            length_before_new_iter=length_before_new_iter,
        )
        loader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
            sampler=sampler,
        )

        loader.collate_fn = self.collate_with_label
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )
        loader.collate_fn = self.collate_with_label
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self._predict_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            multiprocessing_context="fork" if torch.backends.mps.is_available() else None,
        )
        loader.collate_fn = self.collate_features_only
        return loader
