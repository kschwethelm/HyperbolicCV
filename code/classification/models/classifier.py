import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.lorentz.manifold import CustomLorentz
from lib.geoopt.manifolds.stereographic import PoincareBall

from lib.lorentz.layers import LorentzMLR
from lib.poincare.layers import UnidirectionalPoincareMLR

from lib.models.resnet import (
    resnet18,
    resnet50,
    Lorentz_resnet18,
    Lorentz_resnet50
)

EUCLIDEAN_RESNET_MODEL = {
    18: resnet18,
    50: resnet50
}

LORENTZ_RESNET_MODEL = {
    18: Lorentz_resnet18,
    50: Lorentz_resnet50
}

RESNET_MODEL = {
    "euclidean" : EUCLIDEAN_RESNET_MODEL,
    "lorentz" : LORENTZ_RESNET_MODEL,
}

EUCLIDEAN_DECODER = {
    'mlr' : nn.Linear
}

LORENTZ_DECODER = {
    'mlr' : LorentzMLR
}

POINCARE_DECODER = {
    'mlr' : UnidirectionalPoincareMLR
}

class ResNetClassifier(nn.Module):
    """ Classifier based on ResNet encoder.
    """
    def __init__(self, 
            num_layers:int, 
            enc_type:str="lorentz", 
            dec_type:str="lorentz",
            enc_kwargs={},
            dec_kwargs={}
        ):
        super(ResNetClassifier, self).__init__()

        self.enc_type = enc_type
        self.dec_type = dec_type

        self.clip_r = dec_kwargs['clip_r']

        self.encoder = RESNET_MODEL[enc_type][num_layers](remove_linear=True, **enc_kwargs)
        self.enc_manifold = self.encoder.manifold

        self.dec_manifold = None
        dec_kwargs['embed_dim']*=self.encoder.block.expansion
        if dec_type == "euclidean":
            self.decoder = EUCLIDEAN_DECODER[dec_kwargs['type']](dec_kwargs['embed_dim'], dec_kwargs['num_classes'])
        elif dec_type == "lorentz":
            self.dec_manifold = CustomLorentz(k=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            self.decoder = LORENTZ_DECODER[dec_kwargs['type']](self.dec_manifold, dec_kwargs['embed_dim']+1, dec_kwargs['num_classes'])
        elif dec_type == "poincare":
            self.dec_manifold = PoincareBall(c=dec_kwargs["k"], learnable=dec_kwargs['learn_k'])
            self.decoder = POINCARE_DECODER[dec_kwargs['type']](dec_kwargs['embed_dim'], dec_kwargs['num_classes'], True, self.dec_manifold)
        else:
            raise RuntimeError(f"Decoder manifold {dec_type} not available...")
        
    def check_manifold(self, x):
        if self.enc_type=="euclidean" and self.dec_type=="euclidean":
            pass
        elif self.enc_type=="euclidean" and self.dec_type=="lorentz":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.expmap0(F.pad(x, pad=(1,0), value=0))
        elif self.enc_type=="euclidean" and self.dec_type=="poincare":
            x_norm = torch.norm(x,dim=-1, keepdim=True)
            x = torch.minimum(torch.ones_like(x_norm), self.clip_r/x_norm)*x # Clipped HNNs
            x = self.dec_manifold.expmap0(x)
        elif self.enc_type=="lorentz" and self.dec_type=="euclidean":
            x = self.enc_manifold.logmap0(x)[..., 1:]
        elif self.enc_manifold.k != self.dec_manifold.k:
            x =  self.dec_manifold.expmap0(self.enc_manifold.logmap0(x))
        
        return x
    
    def embed(self, x):
        x = self.encoder(x)
        embed = self.check_manifold(x)
        return embed

    def forward(self, x):
        x = self.encoder(x)
        x = self.check_manifold(x)
        x = self.decoder(x)
        return x
        

