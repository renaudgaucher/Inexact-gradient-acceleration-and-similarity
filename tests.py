from byzfl.fed_framework.data_distributor import DataDistributor, CachedDataset
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from byzfl.benchmark.train import dict_datasets
import torch
base = datasets.MNIST(root='/tmp', train=True, download=True, transform=None)
base.targets = torch.tensor(base.targets).long()
train_ds, val_ds = random_split(base, [50000, 10000])

train_ds.dataset = CachedDataset(base, transform=dict_datasets['mnist'][1])
val_ds.dataset = CachedDataset(base, transform=dict_datasets['mnist'][2])

x,y = next(iter(DataLoader(val_ds, batch_size=16)))
print(type(x), x.shape, x.dtype)