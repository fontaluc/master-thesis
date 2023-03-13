from torchvision.datasets import MNIST

dset_train = MNIST("./data", train=True, download=True)
dset_test  = MNIST("./data", train=False, download=True)