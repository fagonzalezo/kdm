import torch.nn as nn

from ..layers import KDMLayer, RBFKernelLayer
from ..utils import dm2discrete, pure2dm


class KDMClassModel(nn.Module):
    """Discriminative KDM classifier.

    Encoder -> pure KDM -> learned joint KDM -> class probabilities.

    For generative co-training, freeze the encoder externally (the caller
    controls `encoder.requires_grad_(False)`) and add
    `-generative * kdm.log_marginal(pure2dm(encoder(x))).mean()` to the
    training-loop loss.

    Trainability of KDM prototypes is controlled by `x_train`, `y_train`,
    and `w_train`.
    """

    def __init__(
        self,
        encoded_size: int,
        dim_y: int,
        encoder: nn.Module,
        n_comp: int,
        sigma: float = 0.1,
        sigma_trainable: bool = True,
        min_sigma: float = 1e-3,
        x_train: bool = True,
        y_train: bool = True,
        w_train: bool = True,
    ):
        super().__init__()
        self.encoded_size = encoded_size
        self.dim_y = dim_y
        self.n_comp = n_comp
        self.encoder = encoder
        self.kernel = RBFKernelLayer(
            sigma=sigma, dim=encoded_size,
            trainable=sigma_trainable, min_sigma=min_sigma,
        )
        self.kdm = KDMLayer(
            kernel=self.kernel,
            dim_x=encoded_size,
            dim_y=dim_y,
            n_comp=n_comp,
            x_train=x_train,
            y_train=y_train,
            w_train=w_train,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        rho_x = pure2dm(encoded)
        rho_y = self.kdm(rho_x)
        return dm2discrete(rho_y)
