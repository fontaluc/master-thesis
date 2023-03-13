from utils import mnist_subset_dset
from torch.utils.data import DataLoader
import torch
from model import CSVAE_classifier_MNIST
from plotting import plot_samples
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
args = parser.parse_args()

_, dset_test = mnist_subset_dset('./data')
test_loader  = DataLoader(dset_test, batch_size=64)

x, y = next(iter(test_loader))
y_CF = 1 - y

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
csvae = CSVAE_classifier_MNIST(m0 = 0, s0 = 1, m1 = 0, s1 = 1)
input_path = args.input
csvae_state = torch.load(input_path + 'csvae.pt', map_location=device)
csvae.load_state_dict(csvae_state)
csvae = csvae.to(device)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[0].set_title('Observation')
plot_samples(axes[0], x)

# Sample w from the prior using the counterfactual label
w = csvae.priorW(y_CF).sample()
z = csvae.posteriorZ(x).sample()
x_CF = csvae.observation_model(w, z).sample()

axes[1].set_title('Counterfactual')
plot_samples(axes[1], x_CF)

plt.savefig(input_path + 'counterfactuals.png')
plt.close(fig)