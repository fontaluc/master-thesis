import math 
import torch
from torch import nn, Tensor
from torch.distributions import Distribution, Normal, ContinuousBernoulli, Categorical
from typing import *
import numpy as np

class ReparameterizedDiagonalGaussian(Distribution):
    """
    A distribution `N(y | mu, sigma I)` compatible with the reparameterization trick given `epsilon ~ N(0, 1)`.
    """
    def __init__(self, mu: Tensor, log_sigma:Tensor):
        assert mu.shape == log_sigma.shape, f"Tensors `mu` : {mu.shape} and ` log_sigma` : {log_sigma.shape} must be of the same shape"
        self.mu = mu
        self.sigma = log_sigma.exp()
        
    def sample_epsilon(self) -> Tensor:
        """`\eps ~ N(0, I)`"""
        return torch.empty_like(self.mu).normal_()
        
    def sample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (without gradients)"""
        with torch.no_grad():
            return self.rsample()
        
    def rsample(self) -> Tensor:
        """sample `z ~ N(z | mu, sigma)` (with the reparameterization trick) """
        return self.mu + self.sigma * self.sample_epsilon() # <- your code
        
    def log_prob(self, z:Tensor) -> Tensor:
        """return the log probability: log `p(z)`"""
        dist = Normal(self.mu, self.sigma) # <- your code
        return dist.log_prob(z)
    
class CSVAE_MNIST(nn.Module):
    """A CSVAE for the MNIST dataset with
    * a Continuous Bernoulli observation model `p_\theta(x | w, z) = CB(x | g_\theta(w, z))`
    * a Gaussian prior `p(z) = N(z | 0, I)` and p(w | y = 1) = N(w | 0, 0.1*I), p(w | y = 0) = N(w | 3, I)
    * a Gaussian posterior `q_\phi(w|x, y) = N(w | \mu(x, y), \sigma(x, y))` and `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    """
    
    def __init__(self, 
                 w_dim: int = 2, 
                 z_dim: int = 2,
                 m0: float = 3, 
                 s0: float = 1, 
                 m1: float = 0, 
                 s1: float = 0.1) -> None:
        super(CSVAE_MNIST, self).__init__()  

        x_dim = 784
        self.w_dim = w_dim
        self.z_dim = z_dim
        y_dim = 1     

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(w|x,y) = N(z | \mu(x,y), \sigma(x,y)), \mu(x,y),\log\sigma(x,y) = h_\phi(x,y)`
        self.encoderW = nn.Sequential(
            nn.Linear(x_dim + y_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*w_dim) 
        )
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderZ = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*z_dim) 
        )
        
        # Generative Model
        # Decode the latent sample `z` and `w` into the parameters of the observation model
        # `p_\theta(x | w, z) = \prod_i B(x_i | g_\theta(w, z))`
        self.decoder = nn.Sequential(
            nn.Linear(w_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim)            
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        ones = torch.ones((1, w_dim))
        self.register_buffer('prior_params_z',  torch.zeros(torch.Size([1, 2*z_dim])))
        self.register_buffer('prior_params_w', torch.cat(
            (
                torch.cat((m0*ones, math.log(s0)*ones), 1), 
                torch.cat((m1*ones, math.log(s1)*ones), 1)
            ), 
            0
          )
        )
      
    def posteriorW(self, x:Tensor, y:Tensor) -> Distribution:
        """return the distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`"""
        
        # compute the parameters of the posterior
        xy = torch.cat((x, y.view(-1, 1)), 1)
        h_xy = self.encoderW(xy)
        mu, log_sigma =  h_xy.chunk(2, dim=-1)
        
        # return a distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        
    def posteriorZ(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoderZ(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorW(self, y, batch_size=1)-> Distribution:
        """return the distribution `p(w|y)`"""
        prior_params = torch.vstack([self.prior_params_w[int(i)] for i in y])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorZ(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params_z.expand(batch_size, *self.prior_params_z.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, w:Tensor, z:Tensor) -> Distribution:
        """return the distribution `p(x|w, z)`"""
        wz = torch.cat((w, z), 1)
        px_logits = self.decoder(wz)
        return ContinuousBernoulli(logits=px_logits, validate_args=False)

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(w|x,y) and q(z|x) (encoder), sample w~q(w|x,y), z~q(z|x) and return the distribution p(x|w,z) (decoder)"""

        # define the posterior q(w|x,y) / encode x,y into q(w|x,y)
        qw = self.posteriorW(x, y)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posteriorZ(x)

        # define the prior p(w|y)
        pw = self.priorW(y, batch_size=x.size(0))
        
        # define the prior p(z)
        pz = self.priorZ(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: w ~ q(w | x, y), z ~ q(z | x)
        w = qw.rsample()
        z = qz.rsample()
        
        # define the observation model p(x|w, z) = B(x | g(w, z))
        px = self.observation_model(w, z)

        return {'px': px, 'pw': pw, 'pz': pz, 'qw': qw, 'qz': qz, 'w': w, 'z': z}

class AUX(nn.Module):
    """
    Adversarial network
    `q_\delta(y|z) = Cat(y|\pi_\delta(z))
    """
    def __init__(self, z_dim: int = 2, num_classes: int = 2) -> None:
      super(AUX, self).__init__()
      self.z_dim = z_dim
      self.num_classes = num_classes
      self.classifier = nn.Sequential(
          nn.Linear(z_dim, 64),
          nn.ReLU(),
          nn.Linear(64, 64),
          nn.ReLU(),
          nn.Linear(64, num_classes)
      )
    def posterior(self, z:Tensor) -> Distribution:
      """return the distribution `q(y|z) = Cat(y|\pi_\delta(x))`"""
      qy_logits = self.classifier(z)
      return Categorical(logits=qy_logits, validate_args=False)

    def forward(self, z) -> Distribution:
      # define the posterior q(y|z)
      qy = self.posterior(z)
      return qy
    
class CSVAE_classifier_MNIST(nn.Module):
    """CSVAE combined with a classifier for the MNIST dataset with
    * a Continuous Bernoulli observation model `p_\theta(x | w, z) = CB(x | g_\theta(w, z))`
    * a Gaussian prior p(w | y = 1) = N(w | 0, 0.1*I), p(w | y = 0) = N(w | 3, I) and `p(z) = N(z | 0, I)`
    * a Gaussian posterior `q_\phi(w|x, y) = N(w | \mu(x, y), \sigma(x, y))` and `q_\phi(z|x) = N(z | \mu(x), \sigma(x))`
    * a Categorical posterior for the classifier `q_\gamma(y|w) = Cat(y|\pi_\gamma(w))`
    """
    
    def __init__(self, 
                 w_dim: int = 2, 
                 z_dim: int = 2, 
                 num_classes: int = 2,
                 m0: float = 3, 
                 s0: float = 1, 
                 m1: float = 0, 
                 s1: float = 0.1) -> None:
        super(CSVAE_classifier_MNIST, self).__init__()  

        x_dim = 784
        self.w_dim = w_dim
        self.z_dim = z_dim
        y_dim = 1     

        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(w|x,y) = N(z | \mu(x,y), \sigma(x,y)), \mu(x,y),\log\sigma(x,y) = h_\phi(x,y)`
        self.encoderW = nn.Sequential(
            nn.Linear(x_dim + y_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*w_dim) 
        )
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoderZ = nn.Sequential(
            nn.Linear(x_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2*z_dim) 
        )
        
        # Generative Model
        # Decode the latent sample `z` and `w` into the parameters of the observation model
        # `p_\theta(x | w, z) = \prod_i B(x_i | g_\theta(w, z))`
        self.decoder = nn.Sequential(
            nn.Linear(w_dim + z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, x_dim)            
        )
        # Classifier
        # Decode the latent sample `w` into the parameters of the classifier
        # `q_\gamma(y|w) = Cat(y|\pi_\gamma(w))`
        self.classifier = nn.Sequential(
            nn.Linear(w_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        ones = torch.ones((1, w_dim))
        self.register_buffer('prior_params_z',  torch.zeros(torch.Size([1, 2*z_dim])))
        self.register_buffer('prior_params_w', torch.cat(
            (
                torch.cat((m0*ones, math.log(s0)*ones), 1), 
                torch.cat((m1*ones, math.log(s1)*ones), 1)
            ), 
            0
          )
        )
      
    def posteriorW(self, x:Tensor, y:Tensor) -> Distribution:
        """return the distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`"""
        
        # compute the parameters of the posterior
        xy = torch.cat((x, y.view(-1, 1)), 1)
        h_xy = self.encoderW(xy)
        mu, log_sigma =  h_xy.chunk(2, dim=-1)
        
        # return a distribution `q(w|x, y) = N(w | \mu(x, y), \sigma(x, y))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
        
    def posteriorZ(self, x:Tensor) -> Distribution:
        """return the distribution `q(z|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoderZ(x)
        mu, log_sigma =  h_x.chunk(2, dim=-1)
        
        # return a distribution `q(z|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorW(self, y, batch_size=1)-> Distribution:
        """return the distribution `p(w|y)`"""
        prior_params = torch.vstack([self.prior_params_w[int(i)] for i in y])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def priorZ(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params_z.expand(batch_size, *self.prior_params_z.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, w:Tensor, z:Tensor) -> Distribution:
        """return the distribution `p(x|w, z)`"""
        wz = torch.cat((w, z), 1)
        px_logits = self.decoder(wz)
        return ContinuousBernoulli(logits=px_logits, validate_args=False)
    
    def posteriorY(self, w: Tensor):
        """return the distribution `p(y|w)`"""
        qy_logits = self.classifier(w)
        return Categorical(logits=qy_logits, validate_args=False)

    def forward(self, x, y) -> Dict[str, Any]:
        """compute the posterior q(w|x,y) and q(z|x) (encoder), sample w~q(w|x,y), z~q(z|x) and return the distribution p(x|w,z) (decoder)"""

        # define the posterior q(w|x,y) / encode x,y into q(w|x,y)
        qw = self.posteriorW(x, y)
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posteriorZ(x)

        # define the prior p(w|y)
        pw = self.priorW(y, batch_size=x.size(0))
        
        # define the prior p(z)
        pz = self.priorZ(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: w ~ q(w | x, y), z ~ q(z | x)
        w = qw.rsample()
        z = qz.rsample()
        
        # define the observation model p(x|w, z) = B(x | g(w, z))
        px = self.observation_model(w, z)

        # define the classifier q(y|w)
        qy = self.posteriorY(w)

        return {'px': px, 'pw': pw, 'pz': pz, 'qw': qw, 'qz': qz, 'w': w, 'z': z, 'qy': qy}

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    return x.view(x.size(0), -1).sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, bx, bw, bz, bh, by):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.bh = bh
        self.by = by
        
    def forward(self, model:nn.Module, aux: nn.Module, x:Tensor, y:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))        
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        m1 = -self.bx*log_px + self.bw*kl_w + self.bz*kl_z

        #Train the encoder to NOT predict y from z
        qy = aux(z) #not detached update the encoder!
        h = qy.entropy() 
        m2 = - self.bh * h # h = - sum(p log p)

        m = m1 + m2

        csvaeLoss = m.mean()

        #Train the aux net to predict y from z
        qy = aux(z.detach()) #detach: to ONLY update the AUX net
        log_qy = qy.log_prob(y)
        exp_log_qy = torch.exp(log_qy)
        n = self.by * exp_log_qy
        auxLoss = - n.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m1': m1, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'h': h, 'qy': exp_log_qy, 'm': m}
            
        return csvaeLoss, auxLoss, diagnostics, outputs

class VI_classifier(nn.Module):
    def __init__(self, bx, bw, bz, by):
        super().__init__()
        self.bx = bx
        self.bw = bw
        self.bz = bz
        self.by = by
        
    def forward(self, model:nn.Module, x:Tensor, y:Tensor) -> Tuple[Tensor, Dict]:
        
        # forward pass through the model
        outputs = model(x, y)
        
        # unpack outputs
        px, pw, pz, qw, qz, w, z, qy = [outputs[k] for k in ["px", "pw", "pz", "qw", "qz", "w", "z", "qy"]]
        
        # evaluate log probabilities
        log_px = reduce(px.log_prob(x))
        log_pw = reduce(pw.log_prob(w))
        log_pz = reduce(pz.log_prob(z))
        log_qw = reduce(qw.log_prob(w))
        log_qz = reduce(qz.log_prob(z))
        log_qy = reduce(qy.log_prob(y))          
        
        # compute the variational lower bound with beta parameters:
        # `m1 = - \beta_1 E_q [ log p(x|z) ] + \beta_2 * D_KL(q(w|x,y) | p(w|y))` + \beta_3 * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(w|x,y) | p(w|y)) = log q(w|x,y) - log p(w|y)` and `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kl_w = log_qw - log_pw
        kl_z = log_qz - log_pz
        
        m = - self.bx*log_px + self.bw*kl_w + self.bz*kl_z - self.by*log_qy
        loss = m.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'m': m, 'log_px':log_px, 'kl_w': kl_w, 'kl_z': kl_z, 'log_qy': log_qy}
            
        return loss, diagnostics, outputs