import os
from typing import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision.utils import make_grid
import wandb

def plot_samples(ax, x):
    x = x.to('cpu')
    nrow = int(np.sqrt(x.size(0)))
    x_grid = make_grid(x.view(-1, 1, 28, 28), nrow=nrow).permute(1, 2, 0)
    ax.imshow(x_grid)
    ax.axis('off')

def plot_2d_w_latents(ax, qw, w, y, m0, s0, m1, s1):
    w = w.to('cpu')
    y = y.to('cpu')
    scale_factor_0 = 2*s0
    scale_factor_1 = 2*s1
    scale_factor = np.where(y == 1, scale_factor_0, scale_factor_1)
    batch_size = w.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior_0 = plt.Circle((m0, m0), scale_factor_0, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior_0)

    prior_1 = plt.Circle((m1, m1), scale_factor_1, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior_1)

    # plot data points
    mus, sigmas = qw.mu.to('cpu'), qw.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor[i] * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(w[:, 0], w[:, 1], color=colors)
    m_min = min(m0, m1)
    m_max = max(m0, m1)
    ax.set_xlim([m_min - 3, m_max + 3])
    ax.set_ylim([m_min - 3, m_max + 3])
    ax.set_aspect('equal', 'box')


def plot_2d_z_latents(ax, qz, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    scale_factor = 2
    batch_size = z.shape[0]
    palette = sns.color_palette()
    colors = [palette[l] for l in y]

    # plot prior
    prior = plt.Circle((0, 0), scale_factor, color='gray', fill=True, alpha=0.1)
    ax.add_artist(prior)

    # plot data points
    mus, sigmas = qz.mu.to('cpu'), qz.sigma.to('cpu')
    mus = [mus[i].numpy().tolist() for i in range(batch_size)]
    sigmas = [sigmas[i].numpy().tolist() for i in range(batch_size)]

    posteriors = [
        plt.matplotlib.patches.Ellipse(mus[i], *(scale_factor * s for s in sigmas[i]), color=colors[i], fill=False,
                                       alpha=0.3) for i in range(batch_size)]
    for p in posteriors:
        ax.add_artist(p)

    ax.scatter(z[:, 0], z[:, 1], color=colors)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_aspect('equal', 'box')


def plot_latents(ax, z, y):
    z = z.to('cpu')
    y = y.to('cpu')
    palette = sns.color_palette()
    colors = [palette[l] for l in y]
    z = TSNE(n_components=2).fit_transform(z)
    ax.scatter(z[:, 0], z[:, 1], color=colors)


def log_csvae_plots(x, y, outputs, training_data, validation_data, prior = [[3, 1], [0, 0.1]], tmp_img="tmp_vae_out.png", figsize=(20, 15), save=False):
    y = y.flatten()
    fig = plt.figure(figsize = figsize)

    # plot CSVAE loss
    ax = fig.add_subplot(3, 4, 1)
    ax.set_title(r'CSVAE loss')
    ax.plot(training_data['m'], label='Training')
    ax.plot(validation_data['m'], label='Validation')
    ax.legend()

    # plot the observation
    ax = fig.add_subplot(3, 4, 2)
    ax.set_title(r'Observation $\mathbf{x}$')
    plot_samples(ax, x)

    # plot AUX posterior
    ax = fig.add_subplot(3, 4, 3)
    ax.set_title(r'$q_\delta(\mathbf{y}|\mathbf{z})$')
    ax.plot(training_data['qy'], label='Training')
    ax.plot(validation_data['qy'], label='Validation')
    ax.legend()

    # plot M1 
    ax = fig.add_subplot(3, 4, 5)
    ax.set_title(r'$\mathcal{M}_1$')
    ax.plot(training_data['m1'], label='Training')
    ax.plot(validation_data['m1'], label='Validation')
    ax.legend()

    # plot NLL
    ax = fig.add_subplot(3, 4, 6)
    ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{w, z})$')
    ax.plot(training_data['log_px'], label='Training')
    ax.plot(validation_data['log_px'], label='Validation')
    ax.legend()

    # plot KL(w)
    ax = fig.add_subplot(3, 4, 7)
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{w}|\mathbf{x}, \mathbf{y})\ |\ p(\mathbf{w}|\mathbf{y})\right)$')
    ax.plot(training_data['kl_w'], label='Training')
    ax.plot(validation_data['kl_w'], label='Validation')
    ax.legend()

    # plot KL(z)
    ax = fig.add_subplot(3, 4, 8)
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    ax.plot(training_data['kl_z'], label='Training')
    ax.plot(validation_data['kl_z'], label='Validation')
    ax.legend()

    # plot H(Y|Z) 
    ax = fig.add_subplot(3, 4, 9)
    ax.set_title(r'$\mathcal{H}(Y|Z)$')
    ax.plot(training_data['h'], label='Training')
    ax.plot(validation_data['h'], label='Validation')
    ax.legend()      

    # plot posterior samples
    ax = fig.add_subplot(3, 4, 10)
    ax.set_title(
        r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{w}, \mathbf{z}), \mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    px = outputs['px']
    x_sample = px.sample().to('cpu')
    plot_samples(ax, x_sample)
    

    # plot the latent samples
    try:
        ax = fig.add_subplot(3, 4, 11)
        w = outputs['w']
        if w.shape[1] == 2:
            ax.set_title(r'Latent Samples $\mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y})$')
            qw = outputs['qw']
            plot_2d_w_latents(ax, qw, w, y, prior)
        else:
            ax.set_title(r'Latent Samples $\mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y})$ (t-SNE)')
            plot_latents(ax, w, y)

        ax = fig.add_subplot(3, 4, 12)
        z = outputs['z']
        if z.shape[1] == 2:
            ax.set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
            qz = outputs['qz']
            plot_2d_z_latents(ax, qz, z, y)
        else:
            ax.set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ (t-SNE)')
            plot_latents(ax, z, y)
        
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    wandb.log({'training': wandb.Image(tmp_img)})
    if not save:
        os.remove(tmp_img)

def log_csvae_classifier_plots(x, y, outputs, training_data, validation_data, m0, s0, m1, s1, tmp_img="tmp_vae_out.png", figsize=(15, 15), save=False):
    y = y.flatten()
    fig = plt.figure(figsize = figsize)

    # plot KL(w)
    ax = fig.add_subplot(3, 3, 1)
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{w}|\mathbf{x}, \mathbf{y})\ |\ p(\mathbf{w}|\mathbf{y})\right)$')
    ax.plot(training_data['kl_w'], label='Training')
    ax.plot(validation_data['kl_w'], label='Validation')
    ax.legend()

    # plot KL(z)
    ax = fig.add_subplot(3, 3, 2)
    ax.set_title(r'$\mathcal{D}_{\operatorname{KL}}\left(q_\phi(\mathbf{z}|\mathbf{x})\ |\ p(\mathbf{z})\right)$')
    ax.plot(training_data['kl_z'], label='Training')
    ax.plot(validation_data['kl_z'], label='Validation')
    ax.legend()

    # plot loss
    ax = fig.add_subplot(3, 3, 3)
    ax.set_title(r'Loss')
    ax.plot(training_data['m'], label='Training')
    ax.plot(validation_data['m'], label='Validation')
    ax.legend()

    # plot the latent samples
    try:
        ax = fig.add_subplot(3, 3, 4)
        w = outputs['w']
        if w.shape[1] == 2:
            ax.set_title(r'Latent Samples $\mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y})$')
            qw = outputs['qw']
            plot_2d_w_latents(ax, qw, w, y, m0, s0, m1, s1)
        else:
            ax.set_title(r'Latent Samples $\mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y})$ (t-SNE)')
            plot_latents(ax, w, y)

        ax = fig.add_subplot(3, 3, 5)
        z = outputs['z']
        if z.shape[1] == 2:
            ax.set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
            qz = outputs['qz']
            plot_2d_z_latents(ax, qz, z, y)
        else:
            ax.set_title(r'Latent Samples $\mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$ (t-SNE)')
            plot_latents(ax, z, y)
        
    except Exception as e:
        print(f"Could not generate the plot of the latent samples because of exception")
        print(e)

    # plot the observation
    ax = fig.add_subplot(3, 3, 6)
    ax.set_title(r'Observation $\mathbf{x}$')
    plot_samples(ax, x)

    # plot classifier posterior
    ax = fig.add_subplot(3, 3, 7)
    ax.set_title(r'$q_\gamma(\mathbf{y}|\mathbf{w})$')
    ax.plot(training_data['log_qy'], label='Training')
    ax.plot(validation_data['log_qy'], label='Validation')
    ax.legend()

    # plot NLL
    ax = fig.add_subplot(3, 3, 8)
    ax.set_title(r'$\log p_\theta(\mathbf{x} | \mathbf{w, z})$')
    ax.plot(training_data['log_px'], label='Training')
    ax.plot(validation_data['log_px'], label='Validation')
    ax.legend()   

    # plot posterior samples
    ax = fig.add_subplot(3, 3, 9)
    ax.set_title(
        r'Reconstruction $\mathbf{x} \sim p_\theta(\mathbf{x} | \mathbf{w}, \mathbf{z}), \mathbf{w} \sim q_\phi(\mathbf{w} | \mathbf{x}, \mathbf{y}), \mathbf{z} \sim q_\phi(\mathbf{z} | \mathbf{x})$')
    px = outputs['px']
    x_sample = px.sample().to('cpu')
    plot_samples(ax, x_sample)

    # display
    plt.tight_layout()
    plt.savefig(tmp_img)
    plt.close(fig)
    wandb.log({'training': wandb.Image(tmp_img)})
    if not save:
        os.remove(tmp_img)
