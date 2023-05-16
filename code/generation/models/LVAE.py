import torch
import torch.nn as nn

from models.VAE_blocks import *

class LVAE(nn.Module):
    """ Implementation of a fully hyperbolic Variational Autoencoder in the lorentz model.
    
    Distribution of latent space = Wrapped normal distribution defined by Nagano et al. (2019).

    Args:
        img_dim: dimensionality of input image (C x H x W)
        enc_layers: Number of encoder convolutional layers
        dec_layers: Number of decoder tranposed convolutional layers
        z_dim: Number of latent dimensions
        initial_filters: Number of output filters of first convolutional layer. Gets doubled with each conv. layer.
        learn_curvature: If curvature of hyperbolic space should be learnable
        enc_K: Encoder curvature
        dec_K: Decoder curvature
    """

    def __init__(self, 
        img_dim, 
        enc_layers, 
        dec_layers, 
        z_dim, 
        initial_filters,
        learn_curvature = False,
        enc_K = 1.0,
        dec_K = 1.0
    ):
        super(LVAE, self).__init__()

        self.encoder = H_Encoder(
            img_dim, 
            enc_layers, 
            z_dim, 
            initial_filters, 
            learn_curvature, 
            enc_K
        )
        self.embedding = H_Embedding(z_dim, share_manifold=self.encoder.manifold)

        self.decoder = H_Decoder(
            img_dim, 
            dec_layers, 
            z_dim, 
            initial_filters*(2**(enc_layers-1)), 
            learn_curvature, 
            dec_K
        )

    def check_k_embed(self, x):
        """ Changes curvature of hyperbolic vectors for embedding layer, if necessary. """
        if self.encoder.manifold.k != self.embedding.manifold.k:
            x = self.encoder.manifold.logmap0(x)
            x = self.embedding.manifold.expmap0(x)
        return x
    
    def check_k_dec(self, x):
        """ Changes curvature of hyperbolic vectors for decoder, if necessary. """
        if self.embedding.manifold.k != self.decoder.manifold.k:
            x = self.embedding.manifold.logmap0(x)
            x = self.decoder.manifold.expmap0(x)
        return x

    def embed(self, x):
        mean, var = self.encoder(x)
        return self.embedding(mean, var)[0]
    
    def generate(self, z):
        """ Generates an image given a latent representation. z has curvature of embedding space. """
        z = self.check_k_dec(z)
        return self.decoder(z)

    def generate_random(self, num_imgs, device):
        """ Generates an image by drawing a latent representation from the prior. """
        z = self.embedding.random_sample(num_imgs, device)
        z = self.check_k_dec(z)
        return self.decoder(z)

    def reconstruct(self, x):
        """ Reconstructs an input image. """
        return self.forward(x)[0]

    def forward(self, x):
        mean, var = self.encoder(x)
        mean_dec = self.check_k_embed(mean)
        z, mean_H, covar, u, v = self.embedding(mean_dec, var)
        z_dec = self.check_k_dec(z)
        x_hat = self.decoder(z_dec)

        return x_hat, z, mean_H, covar, u, v

    def loss(self, x, outputs):
        """ Computes the ELBO loss. """
        x_hat, z, mean_H, covar, u, v = outputs
        rec_loss = 0.5 * torch.sum(torch.square(x-x_hat), dim=(1,2,3)) # MSE
        kl_loss = self.embedding.loss(z, mean_H, covar, u, v)

        return rec_loss, kl_loss