import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Flatten, Sequential

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR
from lib.lorentz.distributions import LorentzWrappedNormal
from lib.lorentz.blocks.layer_blocks import LFC_Block, LConv2d_Block, LTransposedConv2d_Block

from lib.geoopt.manifolds.stereographic import PoincareBall
from lib.poincare.distributions import PoincareWrappedNormal

from lib.Euclidean.blocks.layer_blocks import FC_Block, Conv2d_Block, TransposedConv2d_Block

#################################################
#       Hyperbolic (Lorentz)
#################################################
class H_Encoder(nn.Module):
    """ Implementation of a fully hyperbolic convolutional encoder for embedding an image.
    """
    def __init__(self, 
            img_dim,
            num_layers,
            z_dim,
            initial_filters,
            learn_curvature = False,
            curvature = 1.0
        ):
        super(H_Encoder, self).__init__()

        self.eps = 1e-5

        self.manifold = CustomLorentz(k=curvature, learnable=learn_curvature)
        self.learn_curvature = learn_curvature

        self.z_dim = z_dim

        d,h,w = img_dim
        self.z_h, self.z_w = int(h/(2**num_layers)), int(w/(2**num_layers))

        self.z_filters = int(initial_filters*2**(num_layers-1))
        self.flatten_dim = int(self.z_h*self.z_w*self.z_filters)

        # Convolutional layers
        self.conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i==0:
                in_channels = d+1
            else:
                in_channels = (initial_filters*(2**(i-1))) + 1
            out_channels = (initial_filters*(2**i)) + 1

            self.conv_layers.add_module("Conv_"+str(i), LConv2d_Block(
                manifold=self.manifold, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                bias=True,
                activation=torch.relu,
                normalization="batch_norm")
            )

        # Fully-connected layers
        self.fcMean = LFC_Block(
            manifold=self.manifold, 
            in_features=self.flatten_dim+1, 
            out_features=z_dim+1, 
            bias=True,
            activation=None,
            normalization="None"
        )
        self.fcVar = LFC_Block(
            manifold=self.manifold, 
            in_features=self.flatten_dim+1,
            out_features=z_dim+1, 
            bias=True,
            activation=None,
            normalization="None"
        )
        

    def forward(self, x):
        # project image pixels to hyperbolic space
        x = x.permute(0,2,3,1)
        # -> FROM HERE: CHANNEL LAST!!!
        x = F.pad(x, pad=(1,0), mode="constant", value=0)
        x = self.manifold.projx(x)

        x = self.conv_layers(x)
        x = self.manifold.lorentz_flatten(x)

        # Embed
        mean = self.fcMean(x)
        var = self.fcVar(x)
        var = torch.clamp_min(F.softplus(var[..., 1:]), self.eps)

        return mean, var

class H_Embedding(nn.Module):
    """ Implementation of a hyperbolic embedding layer with wrapped normal distribution.
    """
    def __init__(
            self,
            z_dim,
            learn_curvature = False,
            curvature = 1.0,
            euclidean_input = False,
            share_manifold: CustomLorentz = None
        ):
        super(H_Embedding, self).__init__()

        if share_manifold is None:
            self.manifold = CustomLorentz(k=curvature, learnable=learn_curvature)
        else:
            self.manifold = share_manifold

        self.euclidean_input = euclidean_input

        self.z_dim = z_dim

        self.distr = LorentzWrappedNormal(self.manifold)

    def check_euclidean(self, x):
        if self.euclidean_input:
            x = self.manifold.projx(F.pad(x, pad=(1,0), mode="constant", value=0))
        return x

    def random_sample(self, num_samples, device, mean_H=None, var=None, rescale_var=True):
        """ Draws multiple latent variables from the latent distribution.

        If no mean and variance is given, assume standard normal
        """
        if mean_H is None:
            mean_H = self.manifold.origin((1,self.z_dim+1), device=device)
        if var is None:
            var = torch.ones((1,self.z_dim), device=device)

        covar = self.distr.make_covar(var, rescale=rescale_var)
        samples = self.distr.rsample(mean_H, covar, num_samples).transpose(1,0)[0]

        return samples

    def forward(self, mean, var):
        mean_H = self.check_euclidean(mean)

        # Sample from distribution
        covar = self.distr.make_covar(var, rescale=True)
        z, u, v = self.distr.rsample(mean_H, covar, num_samples=1, ret_uv=True) # Note: Loss is not implemented for multiple samples

        return z, mean_H, covar, u[0], v[0]

    def loss(self, z, mean_H, covar, u, v):
        """ Computes kl divergence between posterior and prior. """

        # Compute density of posterior
        logp_z_posterior = self.distr.log_prob(z, mean_H, covar, u, v)

        # Compute density of prior (rescaled standard normal)
        mean_H_pr = self.manifold.origin(mean_H.shape, device=mean_H.device)
        covar_pr = torch.ones((covar.shape[0], covar.shape[1]), device=covar.device)
        covar_pr = self.distr.make_covar(covar_pr, rescale=True)
        
        logp_z_prior = self.distr.log_prob(z, mean_H_pr, covar_pr)

        # Compute KL-divergence between posterior and prior
        kl_div = logp_z_posterior.view(-1) - logp_z_prior.view(-1)

        return kl_div

class H_Decoder(nn.Module):
    """ Implementation of a fully hyperbolic convolutional decoder for image prediction.

    Takes a latent vector of dimension zDim and generates an image.
    """
    def __init__(
            self, 
            img_dim,
            num_layers, 
            z_dim, 
            initial_filters, 
            learn_curvature = False,
            curvature = 1.0
        ):
        super(H_Decoder, self).__init__()

        self.manifold = CustomLorentz(k=curvature, learnable=learn_curvature)

        d,h,w = img_dim
        self.z_h, self.z_w = 8, 8

        self.initial_filters = initial_filters
        self.flatten_dim = int(self.z_h*self.z_w*initial_filters)

        self.pred_dim = 64

        self.fc1 = LFC_Block(
            manifold=self.manifold, 
            in_features=z_dim+1, 
            out_features=self.flatten_dim+1, 
            bias=True,
            activation=torch.relu,
            normalization="batch_norm"
        )

        self.conv_layers = nn.Sequential()
        for i in range(num_layers-1):
            in_channels = int(initial_filters/(2**(i)))
            out_channels = int(initial_filters/(2**((i+1))))
            self.conv_layers.add_module("TrConv_"+str(i), LTransposedConv2d_Block(
                manifold=self.manifold, 
                in_channels=in_channels+1, 
                out_channels=out_channels+1, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                bias=True,
                activation=torch.relu,
                normalization="batch_norm"
            ))

        self.final_conv = LConv2d_Block( 
            manifold=self.manifold, 
            in_channels=int(initial_filters/2**(num_layers-1))+1, 
            out_channels=self.pred_dim+1, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=True,
        )

        self.predictor = LorentzMLR(num_features=self.pred_dim+1, num_classes=d, manifold=self.manifold)


    def forward(self, z):
        x = self.fc1(z)
        x = self.manifold.lorentz_reshape_img(x, img_dim=[self.z_h, self.z_w, self.initial_filters+1])

        x = self.conv_layers(x)
        x = self.final_conv(x)
        
        x = self.predictor(x)
        x = torch.sigmoid(x)

        x = x.permute(0,3,1,2)

        return x

#################################################
#       Hyperbolic (Poincare)
#################################################
class P_Embedding(nn.Module):
    """ Implementation of a hyperbolic Poincaré embedding layer with wrapped normal distribution.
    """
    def __init__(
            self,
            z_dim,
            learn_curvature = False,
            curvature = 1.0
        ):
        super(P_Embedding, self).__init__()

        self.manifold = PoincareBall(c=curvature, learnable=learn_curvature)

        self.z_dim = z_dim

        self.distr = PoincareWrappedNormal(self.manifold)

    def random_sample(self, num_samples, device, mean_H=None, var=None, rescale_var=True):
        """ Draws multiple latent variables from the latent distribution.

        If no mean and variance is given, assume standard normal
        """
        if mean_H is None:
            mean_H = self.manifold.origin((1,self.z_dim), device=device)
        if var is None:
            var = torch.ones((1,self.z_dim), device=device)

        covar = self.distr.make_covar(var)
        samples = self.distr.rsample(mean_H, covar, num_samples).transpose(1,0)[0]

        return samples

    def forward(self, mean, var):
        mean_H = self.manifold.expmap0(mean)

        # Sample from distribution
        covar = self.distr.make_covar(var)
        z, u, v = self.distr.rsample(mean_H, covar, num_samples=1, ret_uv=True) # Note: Loss is not implemented for multiple samples

        return z, mean_H, covar, u[0], v[0]

    def loss(self, z, mean_H, covar, u, v):
        """ Computes kl divergence between posterior and prior. """

        # Compute density of posterior
        logp_z_posterior = self.distr.log_prob(z, mean_H, covar, u, v)

        # Compute density of prior (rescaled standard normal)
        mean_H_pr = self.manifold.origin(mean_H.shape, device=mean_H.device)
        covar_pr = torch.ones((covar.shape[0], covar.shape[1]), device=covar.device)
        covar_pr = self.distr.make_covar(covar_pr)
        
        logp_z_prior = self.distr.log_prob(z, mean_H_pr, covar_pr)

        # Compute KL-divergence between posterior and prior
        kl_div = logp_z_posterior.view(-1) - logp_z_prior.view(-1)

        return kl_div


#################################################
#       Euclidean
#################################################
class E_Encoder(nn.Module):
    """ Implementation of a convolutional encoder for embedding an image.
    """
    def __init__(self, 
                 img_dim,
                 num_layers,
                 z_dim,
                 initial_filters
                ):
        super(E_Encoder, self).__init__()

        self.eps = 1e-5

        self.z_dim = z_dim

        d,h,w = img_dim
        self.z_h, self.z_w = int(h/(2**num_layers)), int(w/(2**num_layers))

        self.z_filters = int(initial_filters*2**(num_layers-1))
        self.flatten_dim = int(self.z_h*self.z_w*self.z_filters)
    
        # Convolutional layers
        self.conv_layers = Sequential()
        for i in range(num_layers):
            if i==0:
                in_channels = d
            else:
                in_channels = (initial_filters*(2**(i-1)))
            out_channels = (initial_filters*(2**i))

            self.conv_layers.add_module("Conv_"+str(i), Conv2d_Block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                activation=torch.relu, 
                bias=True,
                normalization="batch_norm"
            ))

        # Fully-connected layers
        self.fcMean = FC_Block(
            in_features=self.flatten_dim, 
            out_features=z_dim, 
            activation=None,
            normalization="None"
        )
        self.fcVar = FC_Block(
            in_features=self.flatten_dim,
            out_features=z_dim, 
            activation=None,
            normalization="None"
        )

        self.flatten = Flatten()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)

        # Embed
        mean = self.fcMean(x)
        var = self.fcVar(x)
        var = torch.clamp_min(F.softplus(var), self.eps)

        return mean, var

class E_Embedding(nn.Module):
    """ Implementation of a Euclidean embedding layer with normal distribution. 
    """
    def __init__(self, z_dim):
        super(E_Embedding, self).__init__()

        self.z_dim = z_dim

        self.manifold = None

    def random_sample(self, num_samples, device, mean=None, cov=None):
        """ Draws multiple latent variables from  the latent distribution.

        If no mean and variance is given, assume standard normal
        """
        if mean is None:
            mean = torch.zeros((self.z_dim,), device=device)
        if cov is None:
            cov = torch.eye(self.z_dim, device=device)

        sampler = torch.distributions.MultivariateNormal(mean, cov)

        return sampler.sample((num_samples,))

    def forward(self, mean, var):

        # Reparametrization trick
        epsilon = torch.randn_like(var)
        z = mean + torch.sqrt(var) * epsilon

        return z, mean, var

    def loss(self, mean, var):
        """ Computes KL-divergence for normal posterior and standard normal prior. """
        logvar = torch.log(var)
        kl_div = 0.5 * torch.sum(var + mean**2 - logvar - 1, dim=-1)

        return kl_div


class E_Decoder(nn.Module):
    """ Implementation of a convolutional decoder for image prediction.

    Takes a latent vector of dimension zDim and generates an image.
    """
    def __init__(
            self, 
            img_dim,
            num_layers, 
            z_dim, 
            initial_filters
        ):
        super(E_Decoder, self).__init__()

        d,h,w = img_dim
        self.z_h, self.z_w = 8, 8
        self.initial_filters = initial_filters
        self.flatten_dim = int(self.z_h*self.z_w*initial_filters)

        self.pred_dim = 64

        # Layers
        self.fc1 = FC_Block(
            in_features=z_dim, 
            out_features=self.flatten_dim, 
            activation=True, 
            normalization="batch_norm"
        )

        self.conv_layers = Sequential()
        for i in range(num_layers-1):
            in_channels = int(initial_filters/(2**(i)))
            out_channels = int(initial_filters/(2**((i+1))))
            self.conv_layers.add_module("TrConv_"+str(i), TransposedConv2d_Block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                activation=torch.relu, 
                bias=True,
                normalization="batch_norm"
            ))
        
        self.conv_layers.add_module("FinalConv", Conv2d_Block(
            in_channels=int(initial_filters/2**(num_layers-1)), 
            out_channels=self.pred_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=torch.relu,
            bias=True,
            normalization="batch_norm"
        ))

        self.predictor = nn.Conv2d(in_channels=self.pred_dim, out_channels=d, kernel_size=1, bias=True)
        
    def forward(self, z):
        x = self.fc1(z)
        x = x.view(-1, self.initial_filters, self.z_h, self.z_w)
        
        x = self.conv_layers(x)
        x = self.predictor(x)
    
        x = torch.sigmoid(x)

        return x