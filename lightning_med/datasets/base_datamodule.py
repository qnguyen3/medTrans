import pytorch_lightning as pl
from torchvision import transforms
from abc import ABC, abstractmethod

class BaseDataModule(pl.LightningDataModule):
    """BaseDataModule
        Args:
            train_transform: Transformation for train data
            validation_transform: Transformation for validation data
            test_transform: Transformation for test data
            batch_size: Batch size (default: 1)
            num_workers: num_workers for PyTorch DataLoader (default: 0)
            seed: seed for reproducible purpose (default: 1)
            image_size: image size for transformation (default: 224)
    """
    def __init__(self, data_dir: str = None, 
                train_transform: transforms = None,
                validation_transform: transforms = None,
                test_transform: transforms = None,
                batch_size:int = 1,
                num_workers:int = 0,
                seed:int = 1,
                image_size: int = 224):
        super().__init__()

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            raise ValueError('Missing data_dir. The path to data is required')
        self.train_transforms = train_transform
        self.validation_transform = validation_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.image_size = image_size

    def prepare_data(self, *args, **kwargs):
        return super().prepare_data(*args, **kwargs)

    def setup(self, *args, **kwargs):
        return super().setup(*args, **kwargs)

    def train_dataloader(self, *args, **kwargs):
        return super().train_dataloader(*args, **kwargs)

    def val_dataloader(self, *args, **kwargs):
        return super().val_dataloader(*args, **kwargs)

    def test_dataloader(self, *args, **kwargs):
        return super().val_dataloader(*args, **kwargs)


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass
