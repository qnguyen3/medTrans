|   Backbone    | Image Size |    Pretrain     |epoch  | Val Acc%  |            Plots             |
| :-----------: | :--------: | :-------------: | :----: | :-------: | :--------------------------: |
|   swin_base_patch4_window12_384   |    384     | ImageNet-22K+1K |      |       |                              |
| swin_base_patch4_window7_224 |    224     |   ImageNet-22K+1K   |      |       |                              |
| swin_large_patch4_window12_384  |    384     |   ImageNet-22K+1K   |      |        |                              |
| swin_large_patch4_window7_224  |    224     | ImageNet-22K+1K |      |       |                              |
| swin_small_patch4_window7_224 |    224     |   ImageNet-1K   |      |       |                              |
| swin_tiny_patch4_window7_224 |    224     | ImageNet-1K |      |       |
| swin_base_patch4_window12_384_in22k |    384     | ImageNet-22K |  |  | |
| swin_base_patch4_window7_224_in22k |    224     | ImageNet-22K |  |  | |
| swin_large_patch4_window12_384_in22k |    384     | ImageNet-22K |  |  | |
| swin_large_patch4_window7_224_in22k |    224     | ImageNet-22K |  |  | |
| [vision_transfomer_configs[clickhere]](./medTrans\supervised\models\vision_transformer.py) |     384+224    | ImageNet-22K+1K |  |  | |
| [xcit_configs[clickhere]](./medTrans\supervised\models\xcit.py) |     384+224    | ImageNet-22K+1K |  |  | |
<!-- |  |  |  |  |  |  |  | -->
```
pip install requirements.txt
#test with cifar 10
python test.py --arch [config_names] --pretrained --num_classes 10
# for example, fine-tune with cifar 10 on vision_transformer
python test.py --arch vit_tiny_patch16_224 --pretrained --num_classes 10 --finetune
```