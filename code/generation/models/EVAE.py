import torch
import torch.nn as nn

from .VAE_blocks import H_Embedding, P_Embedding, E_Encoder, E_Decoder, E_Embedding


class EVAE(nn.Module):
    """ Implementation of a Convolutional Variational Autoencoder (CVAE) with diagonal covariance matrix.

    Selection between euclidean latent space (Normal distribution) or hyperbolic latent space (Wrapped normal distribution defined by Nagano et al. (2019))

    Args:
        img_dim: dimensionality of input image (d x H x W)
        enc_layers: Number of encoder convolutional layers
        dec_layers: Number of decoder tranposed convolutional layers
        z_dim: Number of latent dimensions
        initial_filters: Number of output filters of first convolutional layer. Gets doubled with each conv. layer.
        latent_distr: ["euclidean","lorentz","poincare"]
        learn_curvature: Set if curvature of hyperbolic embedding space should be learnable
        embed_K: Initial curvature of hyperbolic embedding space
    """

    def __init__(self, 
            img_dim, 
            enc_layers, 
            dec_layers, 
            z_dim, 
            initial_filters, 
            latent_distr = "euclidean", 
            learn_curvature = False,
            embed_K = 1.0
        ):
        super(EVAE, self).__init__()

        self.latent_distr = latent_distr
        
        self.encoder = E_Encoder(
            img_dim, 
            enc_layers, 
            z_dim, 
            initial_filters
        )

        if self.latent_distr == "euclidean":
            self.embedding = E_Embedding(z_dim=z_dim)
        elif self.latent_distr == "lorentz":
            self.embedding = H_Embedding(z_dim=z_dim, learn_curvature=learn_curvature, curvature=embed_K, euclidean_input=True)
        elif self.latent_distr == "poincare":
            self.embedding = P_Embedding(z_dim=z_dim, learn_curvature=learn_curvature, curvature=embed_K)

        self.decoder = E_Decoder(
            img_dim, 
            dec_layers, 
            z_dim, 
            initial_filters*(2**(enc_layers-1))
        )

    def check_hyperbolic(self, x):
        """ Checks if vectors is hyperbolic and maps it to tangent space, if necessary. """
        if self.latent_distr=="lorentz":
            x = self.embedding.manifold.logmap0(x)[..., 1:]
        elif self.latent_distr=="poincare":
            x = self.embedding.manifold.logmap0(x)
        return x

    def embed(self, x):
        mean, var = self.encoder(x)
        return self.embedding(mean, var)[1]

    def generate(self, z):
        """ Generates an image given a latent representation. z has curvature of embedding space. """
        z = self.check_hyperbolic(z)
        return self.decoder(z)

    def generate_random(self, num_imgs, device):
        """ Generates an image by drawing a latent representation from the prior. """
        z = self.embedding.random_sample(num_imgs, device)
        z = self.check_hyperbolic(z)
        return self.decoder(z)

    def reconstruct(self, x):
        """ Reconstructs an input image. """
        return self.forward(x)[0]

    def forward(self, x):
        mean, var = self.encoder(x)
        outputs = self.embedding(mean, var)
        z = outputs[0]
        z_E = self.check_hyperbolic(z)
        x_hat = self.decoder(z_E)

        if self.latent_distr=="euclidean":
            z, mean, var = outputs
            return x_hat, z, mean, var
        else:
            z, mean_H, var, u, v = outputs
            return x_hat, z, mean_H, var, u, v
            

    def loss(self, x, outputs):
        """ Computes the ELBO loss. """
        x_hat = outputs[0]
        
        rec_loss = 0.5 * torch.sum(torch.square(x-x_hat), dim=(1,2,3)) # MSE
        if self.latent_distr=="euclidean":
            _, _, mean, var = outputs
            kl_loss = self.embedding.loss(mean, var)
        else:
            _, z, mean_H, var, u, v = outputs
            kl_loss = self.embedding.loss(z, mean_H, var, u, v)
            
        return rec_loss, kl_loss
    