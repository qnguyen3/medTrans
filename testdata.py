from numpy import True_
import pytorch_lightning as pl
from lightning_med.datasets import ISIC2019
from lightning_med import GetModel

isic = ISIC2019(data_dir='/host/ubuntu/data/isic2019')
model = GetModel(arch='vit_tiny_patch16_224', pretrained=True, learning_rate=0.01, num_classes=8, fine_tune=True)

trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(model, isic)


