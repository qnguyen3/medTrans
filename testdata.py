from numpy import True_
import pytorch_lightning as pl
from lightning_med.datasets import ISIC2019, ISIC2019_CV
from lightning_med import GetModel
from lightning_med.datasets.utils import KFoldLoop

isic = ISIC2019_CV(data_dir='/host/ubuntu/data/isic2019')
model = GetModel(arch='vit_tiny_patch16_224', pretrained=True, learning_rate=0.01, num_classes=8, fine_tune=True)

trainer = pl.Trainer(gpus=1, max_epochs=5,num_sanity_val_steps=0)
trainer.fit_loop = KFoldLoop(5, trainer.fit_loop, export_path="./")
trainer.fit(model, train_dataloaders=isic)


