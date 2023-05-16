""" Calculates the Frechet Inception Distance (FID) for evaluating image generation models

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a generation model.

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

We adapted the code from https://github.com/mseitzer/pytorch-fid

"""

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import TensorDataset, DataLoader

from .inception import InceptionV3


# ---------------------------------------------------------------------------------------------
# Preloading a few things to make calculation of validation FID while training more efficient
mu_real = None
sigma_real = None
# ---------------------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(dataloader: DataLoader, device: str):
    """ Calculation of the inception activation statistics used by the FID for a set of images.

    Note: Inception model rescales images to 299x299 and normalizes them between -1 and 1 (images have to be between 0 and 1), 
    can be changed in the inception instantion above.

    Args:
        dataloader: PyTorch dataloader containing image data with values between 0 and 1, shape=(d x H x W)
        device: Device for computations (e.g. 'cpu' or 'cuda:0')

    Returns:
        Calculated activation statictics (mean and covariance)
    """

    dtype_before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    inception_model = InceptionV3(resize_input=True, normalize_input=True)
    
    num_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    # Assure everything is in correct state
    inception_model.eval()
    inception_model.to(device)

    # Calculate FID
    act = torch.empty((num_imgs, 2048))

    for i, (x, _) in enumerate(dataloader):
        x = x.to(device).type(torch.float32)
        if x.shape[1] == 1: #Only a single color channel
            x = x.repeat(1, 3, 1, 1)

        pred = inception_model(x)[0]
        act[i*batch_size:(i+1)*batch_size] = pred.detach().cpu().reshape(x.shape[0], -1) # PyTorch does not care if -> :(i+1)*batch_size > tensor size
    
    mu = np.mean(act.numpy(), axis=0)
    sigma = np.cov(act.numpy(), rowvar=False)

    torch.set_default_dtype(dtype_before)

    return mu, sigma


def calculate_FID(dataloader_real: DataLoader, dataloader_gen: DataLoader, device: str, recalc_real_stats: bool=True) -> float:
    """ Calculates FID between real data and generated data.
    
    Args:
        dataloader_real: PyTorch dataloader contraining real-world image data, shape=(d x H x W)
        dataloader_gen: PyTorch dataloader contraining generated image data, shape=(d x H x W)
        device: Device for computations (e.g. 'cpu' or 'cuda:0')
        recalc_real_stats: If activation statistics of real-world images data shall be recalculated

    Returns:
        Calculated FID score
    """
    # Global variables so that activation statistics of real dataset do not have to be recalculated all the time
    global mu_real, sigma_real

    if mu_real is None or sigma_real is None or recalc_real_stats:
        mu_real, sigma_real = calculate_activation_statistics(dataloader_real, device)
    mu_gen, sigma_gen = calculate_activation_statistics(dataloader_gen, device)
       
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    
    return fid_value


@torch.no_grad()
def val_FID(model, img_dim, dataloader_val, batch_size_fid_inception, device):
    """ An efficient function to calculate reconstruction and generation FID in validation step in training.
    
    For reconstruction FID we reconstruct #num_samples validation images and compare them to validation images (biased estimator).

    For generation FID we sample a latent embedding from prior distribution, generate images and compare them to validation images.

    Args:
        model: Generation model instance
        img_dim: Dimensionality of images (d x H x W)
        dataloader_val: PyTorch dataloader containing validation images
        batch_size_fid_inception: Batch size to be used for Inception model prediction
        device: Device for generation model

    Returns:
        Reconstruction FID, Generation FID

    """
    model.eval()
    model.to(device)

    batch_size = dataloader_val.batch_size
    # Remove ddp sampler from dataloader
    dataloader_val = DataLoader(dataloader_val.dataset, batch_size=batch_size)

    # Make new dataloader with different batch size for fid inception model
    dataloader_real = DataLoader(dataloader_val.dataset, batch_size=batch_size_fid_inception)

    # Compute Reconstruction FID
    data_gen = torch.empty((len(dataloader_val.dataset), img_dim[0], img_dim[1], img_dim[2]))

    for i, (x, _) in enumerate(dataloader_val):
        x = x.to(device)
        x_rec = model.reconstruct(x)
        data_gen[i*batch_size:(i+1)*batch_size] = x_rec.detach().cpu() # PyTorch does not care if -> :(i+1)*batch_size > tensor size

    lab = torch.zeros((data_gen.shape[0], 1)) # Add some fake labels to match other dataset
    dataloader_gen = DataLoader(TensorDataset(data_gen, lab), batch_size=batch_size_fid_inception) 
    FID_rec = calculate_FID(dataloader_real, dataloader_gen, device, recalc_real_stats=False)

    # Compute Generation FID
    data_gen = data_gen*0
    
    for i in range(len(dataloader_val)):
        if i==len(dataloader_val)-1: # Last batch may be smaller than batch_size
            num_imgs = len(dataloader_val.dataset) - (len(dataloader_val)-1)*batch_size
        else:
            num_imgs = batch_size
        x_gen = model.generate_random(num_imgs, device)
        data_gen[i*batch_size:(i+1)*batch_size] = x_gen.detach().cpu() # PyTorch does not care if -> :(i+1)*batch_size > tensor size

    dataloader_gen = DataLoader(TensorDataset(data_gen, lab), batch_size=batch_size_fid_inception) 
    FID_gen = calculate_FID(dataloader_real, dataloader_gen, device, recalc_real_stats=False)

    return FID_rec, FID_gen


@torch.no_grad()
def test_FID(model, img_dim, dataloader_val, dataloader_test, batch_size_fid_inception, device):
    """ Calculates reconstruction and generation FID based on test and validation set. 

    For reconstruction FID we reconstruct validation images and compare them to test images.

    For generation FID we sample a latent embedding from prior distribution, generate images and compare them to test images.

    Args:
        model: Generation model instance
        img_dim: Dimensionality of images (d x H x W)
        dataloader_val: PyTorch dataloader containing validation images
        dataloder_test: PyTorch dataloader containing test images
        batch_size_fid_inception: Batch size to be used for Inception model prediction
        device: Device for generation model

    Returns:
        Reconstruction FID, Generation FID
    """
    assert len(dataloader_val.dataset) == len(dataloader_test.dataset), "Test and validation set do not contain the same amount of images"
    model.eval()
    model.to(device)
    
    dataloader_real = DataLoader(dataloader_test.dataset, batch_size=batch_size_fid_inception)
    
    batch_size = dataloader_test.batch_size

    # -------------------------------
    # Compute Reconstruction FID
    # -------------------------------
    print("Calculating reconstruction FID...")
    data_gen = torch.empty((len(dataloader_val.dataset), img_dim[0], img_dim[1], img_dim[2]))

    for i, (x, _) in enumerate(dataloader_val):
        x = x.to(device)
        x_rec = model.reconstruct(x)
        data_gen[i*batch_size:(i+1)*batch_size] = x_rec.detach().cpu() # PyTorch does not care if -> :(i+1)*batch_size > tensor size

    lab = torch.zeros((data_gen.shape[0], 1)) # Add some fake labels to match other dataset
    dataloader_gen = DataLoader(TensorDataset(data_gen, lab), batch_size=batch_size_fid_inception) 
    FID_rec = calculate_FID(dataloader_real, dataloader_gen, device, recalc_real_stats=True)

    # -------------------------------
    # Compute Generation FID
    # -------------------------------
    print("Calculating generation FID...")
    data_gen = data_gen*0
    
    for i in range(len(dataloader_test)):
        if i==len(dataloader_test)-1: # Last batch may be smaller than batch_size
            num_imgs = len(dataloader_test.dataset) - (len(dataloader_test)-1)*batch_size
        else:
            num_imgs = batch_size
        x_gen = model.generate_random(num_imgs, device)
        data_gen[i*batch_size:(i+1)*batch_size] = x_gen.detach().cpu() # PyTorch does not care if -> :(i+1)*batch_size > tensor size

    dataloader_gen = DataLoader(TensorDataset(data_gen, lab), batch_size=batch_size_fid_inception) 
    FID_gen = calculate_FID(dataloader_real, dataloader_gen, device, recalc_real_stats=False)

    
    return FID_rec, FID_gen