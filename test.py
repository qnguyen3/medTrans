from models.swin_transformer import *

vit = swin_base_patch4_window7_224(pretrained=True)
print(vit)