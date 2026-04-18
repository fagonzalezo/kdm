#  Kernel Density Matrices

> **v2.0 — PyTorch port.** The library is now native PyTorch (no Keras / TensorFlow / TFP).
> The original Keras 3 release is preserved on the [`keras-legacy`](https://github.com/fagonzalezo/kdm/tree/keras-legacy)
> branch and tag `v1.0-keras-final`. To keep using it, install the legacy distribution:
>
> ```zsh
> pip install 'kdm<2'
> ```
>
> Three core notebooks (`kdm_classification`, `kdm_regression`, `kdm_density_estimation`) have
> been ported. The other examples in `examples/` are tagged with a banner and still target
> the Keras release.

Kernel Density Matrices (KDMs) are a generalization of density matrices used in quantum mechanics to represent the probabilistic state of a quantum system. KDMs provide a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models.

# Getting Started

Install the PyTorch port from source:

```zsh
pip install git+https://github.com/fagonzalezo/kdm.git
```

Memory-based models additionally require `faiss`:

```zsh
pip install 'kdm-torch[mem]'
```

Check our [examples](https://github.com/fagonzalezo/kdm/tree/main/examples) to see what you can do.

## Quickstart

```python
import torch, torch.nn as nn
from kdm.models import KDMClassModel
from kdm.init import init_kdm_layer

encoder = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 8))
model   = KDMClassModel(encoded_size=8, dim_y=2, encoder=encoder, n_comp=64, sigma=0.5)

# Initialize support points from a small batch of training data
init_kdm_layer(model.kdm, encoder(x_init).detach(), y_init_onehot, init_sigma=True)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for xb, yb in loader:
    probs = model(xb)
    loss  = nn.functional.nll_loss(torch.log(probs + 1e-8), yb)
    opt.zero_grad(); loss.backward(); opt.step()
```
## Paper

> **Kernel Density Matrices for Probabilistic Deep Learning**
> 
> Fabio A. González, Raúl Ramos-Pollán, Joseph A. Gallego-Mejia
> 
> https://doi.org/10.1007/s42484-025-00299-9
> 
> <p align="justify"><b>Abstract:</b> <i>This paper introduces a novel approach to probabilistic deep learning, kernel density matrices, which provide a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. In quantum mechanics, a density matrix is the most general way to describe the state of a quantum system. This work extends the concept of density matrices by allowing them to be defined in a reproducing kernel Hilbert space. This abstraction allows the construction of differentiable models for density estimation, inference, and sampling, and enables their integration into end-to-end deep neural models. In doing so, we provide a versatile representation of marginal and joint probability distributions that allows us to develop a differentiable, compositional, and reversible inference procedure that covers a wide range of machine learning tasks, including density estimation, discriminative learning, and generative modeling. The broad applicability of the framework is illustrated by two examples: an image classification model that can be naturally transformed into a conditional generative model, and a model for learning with label proportions that demonstrates the framework's ability to deal with uncertainty in the training samples.</i></p>

## Citation

If you find this code useful in your research, please consider citing:

```
@article{gonzalez2025kernel,
  title={Kernel density matrices for probabilistic deep learning},
  author={Gonz{\'a}lez, Fabio A and Ramos-Poll{\'a}n, Ra{\'u}l and Gallego, Joseph},
  journal={Quantum Machine Intelligence},
  volume={7},
  number={2},
  pages={94},
  year={2025},
  publisher={Springer}
}
```
