import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torchvision.transforms as transforms
from dataset.dataset import Digits5
import torch
import math

def get_digits5(args, log):
    trainsets, valsets, testsets = [], [], []
    transform_mnist = transforms.Compose([
                transforms.ToTensor(),
            ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
        ])
        
    transform_usps = transforms.Compose(
            [
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
            ]
        )
    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    
    transform_list = [transform_mnist, transform_svhn, transform_usps, transform_synth, transform_mnistm]
    sites = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST_M']
    
    train_sites = sites
    val_sites = sites
    
    for site, transform in zip(sites, transform_list):
        trainset = Digits5(site=site, split='train', transform=transform)
        testset = Digits5(site=site, split='test', transform=transform)
        
        val_len = math.floor(len(trainset)*0.25)
        train_idx = list(range(len(trainset)))[:-val_len]
        val_idx = list(range(len(trainset)))[-val_len:]
        valset   = torch.utils.data.Subset(trainset, val_idx)
        trainset = torch.utils.data.Subset(trainset, train_idx)
        
        log.info(f'[Client {site}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
        trainsets.append(trainset)
        valsets.append(valset)
        testsets.append(testset)

    return train_sites, val_sites, trainsets, valsets, testsets

