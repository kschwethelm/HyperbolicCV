# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
from torch.nn import DataParallel

import configargparse

import random
import numpy as np
import matplotlib.pyplot as plt

from utils.initialize import (
    select_model,
    select_dataset,
    load_model_checkpoint
)

from lib.utils.visualize import (
    visualize_embeddings,
    visualize_generations,
    visualize_reconstructions
)
from train import test


def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='VAE Test Parser', add_help=True)

    parser.add_argument('-c', '--config_file', required=False, default="H-VAE/code/experiments/CelebA/H-VAE.txt", is_config_file=True, type=str, 
                        help="Path to config file.")
    
    # Modes
    parser.add_argument('--mode', default="test_FID", type=str, 
                        choices=[
                            "test_FID",
                            "visualize_embeddings",
                            "generate",
                            "reconstruct"
                        ],
                        help = "Select the testing mode.")
    
    # Output settings
    parser.add_argument('--output_dir', default=None, type=str, 
                        help = "Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0", type=lambda s: [str(item) for item in s.replace(' ','').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"], 
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str, 
                        help = "Path to model checkpoint.")

    # Testing parameters
    parser.add_argument('--batch_size', default=128, type=int, 
                        help="Training batch size.")
    parser.add_argument('--batch_size_test', default=128, type=int, 
                        help="Training batch size.")
    parser.add_argument('--batch_size_fid_inception', default=512, type=int, 
                        help = "Inception model is needed to calculate FID-score. Give batch size for it.")
    
    # General VAE hyperparameters
    parser.add_argument('--model', default='L-VAE', type=str, choices=["L-VAE", "EL-VAE", "EP-VAE", "E-VAE"], 
                        help = "Select a model architecture.")
    parser.add_argument('--enc_layers', default=4, type=int, 
                        help = "Number of convolutional layers in encoder.")
    parser.add_argument('--dec_layers', default=3, type=int, 
                        help = "Number of transposed convolutional layers in decoder.")
    parser.add_argument('--z_dim', default=128, type=int, 
                        help = "Dimensionality of latent space.")
    parser.add_argument('--initial_filters', default=64, type=int, 
                        help = "Number of output filters of first convolutional layer. Gets doubled with each conv. layer.")

    # Hyperbolic manifold settings
    parser.add_argument('--learn_K', action='store_true',
                        help="Placeholder... Not needed here.")
    parser.add_argument('--embed_K', default=1.0, type=float, 
                        help = "Initial curvature of embedding space (geoopt.K=-1/K).")

    # HVAE
    parser.add_argument('--enc_K', default=1.0, type=float, 
                        help = "Initial curvature of encoder space (geoopt.K=-1/K).")
    parser.add_argument('--dec_K', default=1.0, type=float, 
                        help = "Initial curvature of decoder space (geoopt.K=-1/K).")

    # Dataset settings
    parser.add_argument('--dataset', default='CIFAR-10', type=str, choices=["MNIST", "CIFAR-10", "CIFAR-100", "CelebA"], 
                        help = "Select a dataset.")

    args, _ = parser.parse_known_args()

    return args

def main(args):
    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, img_dim = select_dataset(args)

    print("Creating model...")
    model = select_model(img_dim, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model = load_model_checkpoint(model, args.load_checkpoint)
    else:
        print("No model checkpoint given. Using random weights.")

    model = DataParallel(model, device_ids=args.device)
    model.eval()

    if args.mode=="test_FID":
        print("Testing FID of model...")
        test(model, train_loader, val_loader, test_loader, img_dim, device, args)
        
    elif args.mode=="visualize_embeddings":
        print("Visualizing embedding space of model...")
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            output_path = os.path.join(args.output_dir, "embeddings.png")
        else:
            output_path = "embeddings.png"
        save_embeddings(model, train_loader, output_path, device)

    elif args.mode=="generate":
        print("Visualizing model generations...")
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            output_path = os.path.join(args.output_dir, "generations.png")
        else:
            output_path = "generations.png"
        save_generations(model, output_path, device)

    elif args.mode=="reconstruct":
        print("Visualizing model generations...")
        if args.output_dir is not None:
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            output_path = os.path.join(args.output_dir, "reconstructions.png")
        else:
            output_path = "reconstructions.png"
        save_reconstructions(model, train_loader, output_path, device)

    else:
        print(f"Mode {args.mode} not implemented yet.")
    
    print("Finished!")

@torch.no_grad()
def save_embeddings(model, data_loader, output_path, device):
    fig = visualize_embeddings(model, data_loader, device, model.module.embedding.manifold, args.model=="EP-VAE")
    print(f"Saving embeddings to {output_path}...")
    fig.savefig(output_path)
    plt.close(fig)

@torch.no_grad()
def save_generations(model, output_path, device):
    fig = visualize_generations(model, device, num_imgs_per_axis=5)
    print(f"Saving generations to {output_path}...")
    fig.savefig(output_path)
    plt.close(fig)

@torch.no_grad()
def save_reconstructions(model, data_loader, output_path, device):
    fig = visualize_reconstructions(model, data_loader, device, num_imgs=5)
    print(f"Saving reconstructions to {output_path}...")
    fig.savefig(output_path)
    plt.close(fig)

# ----------------------------------
if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    main(args)
    