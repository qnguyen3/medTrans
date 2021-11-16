import argparse

from pytorch_lightning import loggers
from medTrans import GetModel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch
import pytorch_lightning as pl

def get_args_parser():
    parser = argparse.ArgumentParser('medTrans', add_help=False)
    parser.add_argument('--arch', default='vit_tiny_patch16_224', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--gpu_mode', default='ddp', type=str, choices=['ddp', 'dp', 'cpu'])
    return parser

def main(args):
    
    model = GetModel(arch=args.arch, pretrained=args.pretrained, learning_rate=args.lr, num_classes=args.num_classes)

    print(model)

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, trainloader, testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('medTrans', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
