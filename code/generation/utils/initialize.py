import torch
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import StepLR

from models.LVAE import LVAE
from models.EVAE import EVAE

def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(img_dim, args):
    """ Selects and sets up an available model and returns it. """

    if args.model == "L-VAE":
        model = LVAE(
            img_dim, 
            args.enc_layers, 
            args.dec_layers, 
            args.z_dim, 
            args.initial_filters, 
            args.learn_K,
            args.enc_K,
            args.dec_K
        )
    elif args.model == "EL-VAE":
        model = EVAE(
            img_dim, 
            args.enc_layers, 
            args.dec_layers, 
            args.z_dim, 
            initial_filters=args.initial_filters, 
            latent_distr="lorentz", 
            learn_curvature=args.learn_K,
            embed_K=args.embed_K
        )
    elif args.model == "EP-VAE":
        model = EVAE(
            img_dim, 
            args.enc_layers, 
            args.dec_layers, 
            args.z_dim, 
            initial_filters=args.initial_filters, 
            latent_distr="poincare", 
            learn_curvature=args.learn_K,
            embed_K=args.embed_K
        )
    elif args.model == "E-VAE":
        model = EVAE(
            img_dim, 
            args.enc_layers, 
            args.dec_layers, 
            args.z_dim, 
            initial_filters=args.initial_filters,
            latent_distr="euclidean"
        )
    else:
        raise "Model not found. Wrong model in configuration... -> " + args.model

    return model

def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    if args.use_lr_scheduler:
        lr_scheduler = StepLR(
            optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def select_dataset(args):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """
    
    if args.dataset == 'MNIST':
        
        train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32), antialias=None)
        ])

        train_data = datasets.MNIST('data', train=True, download=True, transform=train_transform)
        train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.MNIST('data', train=False, download=True, transform=test_transform)

        img_dim = [1, 32, 32]

    elif args.dataset == 'CIFAR-10':
        transform=transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_data, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR10('data', train=False, download=True, transform=transform)

        img_dim = [3, 32, 32]

    elif args.dataset == 'CIFAR-100':
        transform=transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(train_data, [40000, 10000], generator=torch.Generator().manual_seed(1))
        test_set = datasets.CIFAR100('data', train=False, download=True, transform=transform)

        img_dim = [3, 32, 32]

    elif args.dataset == 'CelebA':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64,64), antialias=None)
        ])

        train_set = datasets.CelebA('data', split='train', download=True, transform=transform)
        val_set = datasets.CelebA('data', split='valid', download=True, transform=transform)
        test_data = datasets.CelebA('data', split='test', download=True, transform=transform)
        # Test and val set should have same size for FID calculation
        test_set, _ = torch.utils.data.random_split(test_data, [len(val_set), len(test_data)-len(val_set)], generator=torch.Generator().manual_seed(1))

        img_dim = [3, 64, 64]

    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)
    
    # Dataloader
    train_loader = DataLoader(train_set, 
        batch_size=args.batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=True,
        drop_last=(args.dataset=='CelebA')
    )
    val_loader = DataLoader(val_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    )
    test_loader = DataLoader(test_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False,
    ) 
    
    return train_loader, val_loader, test_loader, img_dim