import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformer.SubLayers import MultiHeadAttentionNew, PositionwiseFeedForward

def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    #h 600 l 200 i 20108
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    
class newEncoder(nn.Module):
    #h 600 l 200 i 20108
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(newEncoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        return self.fc_mu(h1), self.fc_logvar(h1)

class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, eps=1e-1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        return self.fc(x)

class TEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, input_dim, latent_dim, n_layers=1, d_model=200, n_head=6, d_k=64, d_v=64, dropout=0.5):

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.slf_attn = MultiHeadAttentionNew(n_head, input_dim, d_k, d_v, dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc = nn.Linear(input_dim, latent_dim, bias=False)

    def forward(self, src_seq, return_attns=False):
        # -- Forward

        #enc_output = self.dropout(self.position_enc(src_seq))
        #enc_output = self.layer_norm(enc_output)
        enc_output, enc_slf_attn = self.slf_attn(src_seq, src_seq, src_seq)

        #enc_output = self.pos_ffn(enc_output)
        enc_output = self.fc(enc_output)

        if return_attns:
            return enc_output, enc_slf_attn
        return enc_output

        

class VAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        #self.newencoder = newEncoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.tecoder = TEncoder(latent_dim+input_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def makenewin(self, latent, deout):
        newinput = torch.cat((latent, deout),1)
        return newinput

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z, dropout_rate=dropout_rate)
        newlatent = self.makenewin(z, x_pred)
        x_pred_p = self.tecoder(newlatent)

        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            """mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)"""
            
            if beta:
                mll = (F.log_softmax(x_pred_p, dim=-1) * user_ratings).sum(dim=-1).mean()
                negative_elbo = -mll
            else:
                mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
                kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
                negative_elbo = -(mll - kld)

            return negative_elbo
            
        else:
            return x_pred_p

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
