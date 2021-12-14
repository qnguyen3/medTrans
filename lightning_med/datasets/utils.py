import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import csv
import os
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from .base_datamodule import BaseKFoldDataModule

class EnsembleVotingModel(pl.LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        logits = torch.stack([m(batch[0]) for m in self.models]).mean(0)
        loss = F.cross_entropy(logits, batch[1])
        self.log("test_loss", loss)

class KFoldLoop(Loop):
    def __init__(self, num_folds: int, fit_loop: FitLoop, export_path: str):
        super().__init__()
        self.num_folds = num_folds
        self.fit_loop = fit_loop
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.accelerator.setup_optimizers(self.trainer)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.training_type_plugin.connect(voting_model)
        self.trainer.training_type_plugin.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

### Will be refractor
class ISIC2019Dataset(Dataset):
    def __init__(self, csv_file: str, 
                data_dir: str, 
                transforms: transforms=None):
        
        self.images = read_labels_csv(csv_file)
        self.data_dir = os.path.join(data_dir, 'ISIC_2019_Training_Input')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, target = self.images[index]
        image = Image.open(os.path.join(self.data_dir,image_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = np.argmax(target)

        return image, target

def read_labels_csv(file,header=True):
    images = []
    num_categories = 0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class ISIC2019Dataset_CV(Dataset):
    def __init__(self, csv_file: str, 
                data_dir: str, 
                transforms: transforms=None):
        
        self.images_name, self.targets = read_image_csv(csv_file)
        self.data_dir = os.path.join(data_dir, 'ISIC_2019_Training_Input')
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name, target = self.images_name[index], self.targets[index]
        image = Image.open(os.path.join(self.data_dir,image_name+'.jpg')).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = np.argmax(target)

        return image, target

def read_image_csv(csv_path = None):
    dataset = pd.read_csv(filepath_or_buffer=csv_path)
    image_name = dataset.iloc[1:,0].to_numpy()
    target_set = dataset.iloc[:, 1:].drop(columns='UNK').to_numpy()

    return (image_name, target_set)

