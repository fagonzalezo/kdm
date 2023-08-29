#  Kernel Density Matrices

Kernel Density Matrices (KDMs) are a generalization of density matrices used in quantum mechanics to represent the probabilistic state of a quantum system. KDMs provide a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models. 

## Paper

> **Kernel Density Matrices for Probabilistic Deep Learning**
> 
> Fabio A. González, Raúl Ramos-Pollán, Joseph A. Gallego-Mejia
> 
> https://arxiv.org/abs/2305.18204
> 
> <p align="justify"><b>Abstract:</b> <i>This paper introduces a novel approach to probabilistic deep learning, kernel density matrices, which provide a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. In quantum mechanics, a density matrix is the most general way to describe the state of a quantum system. This work extends the concept of density matrices by allowing them to be defined in a reproducing kernel Hilbert space. This abstraction allows the construction of differentiable models for density estimation, inference, and sampling, and enables their integration into end-to-end deep neural models. In doing so, we provide a versatile representation of marginal and joint probability distributions that allows us to develop a differentiable, compositional, and reversible inference procedure that covers a wide range of machine learning tasks, including density estimation, discriminative learning, and generative modeling. The broad applicability of the framework is illustrated by two examples: an image classification model that can be naturally transformed into a conditional generative model, and a model for learning with label proportions that demonstrates the framework's ability to deal with uncertainty in the training samples.</i></p>

## Citation

If you find this code useful in your research, please consider citing:

```
@misc{gonzalez2023quantum,
      title={Kernel Density Matrices for Probabilistic Deep Learning}, 
      author={Fabio A. González and Raúl Ramos-Pollán and Joseph A. Gallego-Mejia},
      year={2023},
      eprint={2305.18204},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
