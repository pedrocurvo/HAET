"""
Multi-Layer Perceptron (MLP) implementation with optional residual connections.

This module provides a flexible MLP implementation that can be configured with
different activation functions and an arbitrary number of hidden layers.
"""

import torch.nn as nn

# Dictionary mapping activation function names to their PyTorch implementations
ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable hidden layers and optional residual connections.

    This MLP consists of an input projection, multiple hidden layers with
    the same dimension, and an output projection. Residual connections can be
    added between hidden layers for improved gradient flow.

    Attributes:
        n_input (int): Input feature dimension
        n_hidden (int): Hidden layer dimension
        n_output (int): Output feature dimension
        n_layers (int): Number of hidden layers
        res (bool): Whether to use residual connections
        linear_pre (nn.Sequential): Input projection with activation
        linear_post (nn.Linear): Output projection
        linears (nn.ModuleList): List of hidden layers with activations
    """

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        """Initialize the MLP.

        Args:
            n_input: Input feature dimension
            n_hidden: Hidden layer dimension
            n_output: Output feature dimension
            n_layers: Number of hidden layers
            act: Activation function name (must be in ACTIVATION dictionary)
            res: Whether to use residual connections between hidden layers

        Raises:
            NotImplementedError: If the activation function is not supported
        """
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(n_hidden, n_hidden), act())
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape [..., n_input]

        Returns:
            Output tensor of shape [..., n_output]
        """
        # Input projection
        x = self.linear_pre(x)

        # Hidden layers with optional residual connections
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x  # Residual connection
            else:
                x = self.linears[i](x)

        # Output projection
        x = self.linear_post(x)
        return x
