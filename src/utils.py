from collections import defaultdict
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from hydra.utils import get_original_cwd
import matplotlib.pyplot as plt

def mnist_subset_dset(data_path, classes = [4, 9]):
    """
    Return train and test datasets for MNIST with only a subset of the classes
    """
    # Flatten the images into a vector
    flatten = lambda x: ToTensor()(x).view(28**2)

    dset_train = MNIST(data_path, train=True,  transform=flatten, download=False)
    dset_test  = MNIST(data_path, train=False, transform=flatten, download=False)

    classDict = dict(zip(classes, range(len(classes))))

    N_train = len(dset_train)
    indices = torch.zeros(N_train, dtype = bool)
    for c in classes:
        idx = (dset_train.targets == c)
        indices = indices | idx

    dset_train.data, dset_train.targets = dset_train.data[indices], torch.tensor([classDict[cl.item()] for cl in dset_train.targets[indices]])

    N_test = len(dset_test)
    indices = torch.zeros(N_test, dtype = bool)
    for c in classes:
        idx = (dset_test.targets == c)
        indices = indices | idx

    dset_test.data, dset_test.targets = dset_test.data[indices], torch.tensor([classDict[cl.item()] for cl in dset_test.targets[indices]])

    return dset_train, dset_test

def project(ax, y, attr, num_classes=2, title=None):
    """
    2D scatter plot
    """
    attr = attr.to('cpu')
    if title is not None:
        ax.set_title(title)
    for c in range(num_classes):
        ax.scatter(attr[y == c, 0], attr[y == c, 1], s=1, alpha=0.3)

def visualize_latent(dset_test, csvae, device):
    """
    Visualize latent samples for the test set
    """
    xs = dset_test.data.view(-1, 784).float()
    ys = dset_test.targets

    with torch.no_grad():
        outputs = csvae(xs.to(device), ys.to(device))

    px, pw, pz, qw, qz, ws, zs = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    project(axes[0], ys, ws, title='w projection')
    project(axes[1], ys, zs, title='z projection')

    plt.savefig('latent.png')
    plt.close(fig)