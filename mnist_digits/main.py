import torch
import torch.utils.data
from model import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from tqdm import trange
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from functools import reduce 
from torchvision.utils import make_grid

torch.autograd.set_detect_anomaly(True)

def KL(mu1, logvar1, mu2, logvar2):
    std1 = torch.exp(0.5 * logvar1)
    std2 = torch.exp(0.5 * logvar2)
    return torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)


def project(ax, ys, attr, title=None):
    # scatter plot
    # plt.figure()
    colors = np.zeros((ys.shape[0], 3))
    colors[ys == 1] = np.array((1., 0, 0))
    colors[ys == 0] = np.array((0, 0, 1.))
    if title is not None:
        ax.set_title(title)
    ax.scatter(attr[:, 0], attr[:, 1], s=1, c=colors, alpha=0.3)

def stratified_sampler(labels):
    """Sampler that only picks datapoints corresponding to the specified classes"""
    (indices,) = np.where(reduce(lambda x, y: x | y, [labels.numpy() == i for i in classes]))
    indices = torch.from_numpy(indices)
    return SubsetRandomSampler(indices)

def plot_samples(ax, x):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')

if __name__ == "__main__":

    flatten = lambda x: ToTensor()(x).view(28**2)
    dset_train = MNIST("./", train=True,  transform=flatten, download=True)
    classes = [4, 9]
    batch_size = 64
    trainloader = DataLoader(dset_train, batch_size=batch_size,
                            sampler=stratified_sampler(dset_train.train_labels))

    x_train = dset_train.data.view(-1, 784).float()
    targets_train = dset_train.targets
    mask = np.isin(targets_train, classes)
    xs = x_train[mask]
    ys = (targets_train[mask] == classes[0]).type(torch.int32)

    decoder_x = DecoderX(4, 784)
    encoder_w = EncoderW(785, 2)
    encoder_z = EncoderZ(784, 2)
    decoder_y = DecoderY(2)

    optimizer1 = torch.optim.Adam(chain(decoder_x.parameters(), 
        encoder_w.parameters(), 
        encoder_z.parameters()), lr=1e-3)
    optimizer2 = torch.optim.Adam(decoder_y.parameters(), lr=1e-3)

    with trange(100) as t:
        for i in t:
            t.set_description('Epoch %d' % i)
            for x, y in trainloader:
                mu_z, logvar_z, z = encoder_z(x)
                mu_w, logvar_w, w = encoder_w(x, y.unsqueeze(-1).float())

                mu_x, logvar_x, pred_x = decoder_x(w, z)

                kl1 = KL(mu_w, logvar_w, torch.zeros_like(mu_w), torch.ones_like(logvar_w) * np.log(0.01))
                kl0 = KL(mu_w, logvar_w, torch.ones_like(mu_w) * 3., torch.zeros_like(logvar_w))

                #Train the encoder to NOT predict y from z
                # pred_y = decoder_y(z) #not detached update the encoder!
                loss1 = (20. * torch.sum((x - mu_x) ** 2, -1)
                    + 1. * torch.where(y == 1, kl1, kl0)
                    + 0.2 * KL(mu_z, logvar_z, torch.zeros_like(mu_z), torch.zeros_like(logvar_z))).sum()
                    # + 10. * torch.sum(pred_y * torch.log(pred_y), -1)).sum()  # maximize entropy, enforce uniform distribution
                
                #Train the aux net to predict y from z
                # pred_y = decoder_y(z.detach()) #detach: to ONLY update the AUX net
                # loss2 = (1. * torch.where(y == 1, -torch.log(pred_y[:, 1]), -torch.log(pred_y[:, 0]))).sum()

                optimizer1.zero_grad()
                loss1.backward()
                optimizer1.step()
                
                # optimizer2.zero_grad()
                # loss2.backward()
                # optimizer2.step()

                loss = loss1 

            t.set_postfix(loss=loss.item())#, y_max=pred_y.max().item(), y_min=pred_y.min().item())

    # reconstruction
    with torch.no_grad():
        zs, _, _ = encoder_z(xs)
        ws, _, _ = encoder_w(xs, ys.unsqueeze(-1).float())

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].set_title('Observation')
    plot_samples(axes[0, 0], x)

    axes[0, 1].set_title('Reconstruction')
    plot_samples(axes[0, 1], pred_x)

    project(axes[1, 0], ys, ws, 'w projection')
    project(axes[1, 1], ys, zs, 'z projection')

    plt.show()
    #plt.savefig('figures/mnist.png')