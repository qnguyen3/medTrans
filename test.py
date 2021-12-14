import argparse

from pytorch_lightning import loggers
import lightning_med as lm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch
import pytorch_lightning as pl
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    

def get_args_parser():
    parser = argparse.ArgumentParser('medTrans', add_help=False)
    parser.add_argument('--arch', default='vit_tiny_patch16_224', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--gpu_mode', default='ddp', type=str, choices=['ddp', 'dp', 'cpu'])
    return parser

def main(args):
    
    model = lm.GetModel(arch=args.arch, pretrained=args.pretrained, learning_rate=args.lr, num_classes=args.num_classes, fine_tune=args.finetune)
    print('finetune: {args.finetune}\n')
    print(model)
    count_parameters(model)

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
    testset, batch_size=256, shuffle=False, num_workers=12)

    trainer = pl.Trainer(gpus=1, max_epochs=args.epoch)
    trainer.fit(model, trainloader)
    trainer.test(testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('medTrans', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
