
import torch
from torch import nn
import torch.nn.functional as F
from mmdyn.pytorch.models.vae import VAE, MVAE, Swish
from mmdyn.pytorch import config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_model(model_name, cross_modal=False, **kwargs):
    assert (model_name in config.MODELS), "Model is not implement yet"

    if 'mvae' in model_name and cross_modal:
        model = MVAE(**kwargs)
    elif 'vae' in model_name:
        assert not cross_modal, "VAE does not work with cross modal inputs."
        model = VAE(**kwargs)
    elif 'regressor' in model_name:
        model = Regressor(**kwargs)
    else:
        exit("The model and modality combination is not valid.")
    return model


class Regressor(nn.Module):

    def __init__(self, out_dim=7, conditional=False, num_classes=None):
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes

        # DCGAN style
        cnn_features_out = 256 * 5 * 5
        cnn_features_comp = 512 + self.conditional * self.num_classes
        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(cnn_features_out, 512),
            Swish(),
            nn.Dropout(p=0.1),
        )
        self.out_net = nn.Sequential(
            nn.Linear(cnn_features_comp, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x, c=None):
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)

        x = self.fc_net(x)

        if self.conditional:
            if c.dim() == 1:
                c = c.unsqueeze(1)
            x = torch.cat((x, c.float()), dim=-1)

        out = self.out_net(x)

        return out
