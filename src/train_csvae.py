from collections import defaultdict
from model import *
from utils import *
import hydra
import wandb
import torch
from torch.utils.data import DataLoader
from plotting import log_csvae_plots
from tqdm import trange

def train(train_loader, training_data, csvae, aux, optimizerCSVAE, optimizerAUX, vi, device):
    training_epoch_data = defaultdict(list)
    csvae.train()
    
    # Go through each batch in the training dataset using the loader
    # Note that y is not necessarily known as it is here
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvaeLoss, auxLoss, diagnostics, outputs = vi(csvae, aux, x, y)
        
        optimizerCSVAE.zero_grad()
        csvaeLoss.backward()
        optimizerCSVAE.step()

        optimizerAUX.zero_grad()
        auxLoss.backward()
        optimizerAUX.step()
        
        # gather data for the current batch
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
            
    # gather data for the full epoch
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    
    return training_data, csvaeLoss, auxLoss

def eval(test_loader, validation_data, csvae, aux, vi, device):
    # Evaluate on a single batch, do not propagate gradients
    with torch.no_grad():
        csvae.eval()
        
        # Just load a single batch from the test loader
        x, y = next(iter(test_loader))
        x = x.to(device)
        y = y.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        csvaeLoss, auxLoss, diagnostics, outputs = vi(csvae, aux, x, y)
        
        # gather data for the validation step
        for k, v in diagnostics.items():
            validation_data[k] += [v.mean().item()]
    
    return validation_data, csvaeLoss, auxLoss, x, y, outputs

@hydra.main(
    version_base=None, config_path="../config", config_name="default_config.yaml"
)
def main(cfg):
    wandb.init(project="thesis")

    hparams = cfg.experiment  
    classes = hparams['classes']
    dset_train, dset_test = mnist_subset_dset(f"{get_original_cwd()}/data", classes)  
    batch_size = hparams['batch_size']
    train_loader = DataLoader(dset_train, batch_size=batch_size)
    test_loader  = DataLoader(dset_test, batch_size=batch_size)

    torch.manual_seed(hparams["seed"])

    m0 = hparams['m0']
    s0 = hparams['s0']
    m1 = hparams['m1']
    s1 = hparams['s1']
    csvae = CSVAE_classifier_MNIST(m0=m0, s0=s0, m1=m1, s1=s1)
    aux = AUX()

    # Evaluator: Variational Inference
    bx = hparams['bx']
    bw = hparams['bw']
    bz = hparams['bz']
    bh = hparams['bh']
    by = hparams['by']
    vi = VariationalInference(bx, bw, bz, bh, by)

    # The Adam optimizer works really well with VAEs.
    lr = hparams['lr']
    optimizerCSVAE = torch.optim.Adam(csvae.parameters(), lr=lr)
    optimizerAUX = torch.optim.Adam(aux.parameters(), lr=lr)

    # define dictionary to store the training curves
    training_data = defaultdict(list)
    validation_data = defaultdict(list)

    epoch = 0
    num_epochs = hparams['epochs']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    csvae = csvae.to(device)
    aux = aux.to(device)

    save = False # don't save temporary logging plots locally

    # training..
    with trange(num_epochs) as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)
            training_data, _, _ = train(train_loader, training_data, csvae, aux, optimizerCSVAE, optimizerAUX, vi, device)
            validation_data, csvae_valid_loss, aux_valid_loss, x, y, outputs = eval(test_loader, validation_data, csvae, aux, vi, device)
            
            if epoch == num_epochs - 1:
                save = True # save final plot locally
            # Log the training curves, the test observations, reconstructions and latent samples
            log_csvae_plots(x, y, outputs, training_data, validation_data, m0=m0, s0=s0, m1=m1, s1=s1, save=save)

            t.set_postfix(csvaeLoss=csvae_valid_loss.item(), auxLoss=aux_valid_loss.item())

    torch.save(csvae.state_dict(), 'csvae_mnist.pt')
    torch.save(aux.state_dict(), 'aux_mnist.pt')

    visualize_latent(dset_test, csvae, device)
    

if __name__ == "__main__":   
    main()