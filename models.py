"""
Single-hidden-layer Expectile Regression Neural Network (ERNN).

Architecture (eq. 2 of paper):
    f(x; theta) = sum_j w_j^(o) * sigma(sum_l w_lj^(h) * x_l + b_j^(h)) + b^(o)

Parameter vector ordering (matches paper exactly):
    theta = (vec(W^(h)), w^(o), b^(h), b^(o))

where:
    W^(h) in R^{p x J}  : input-to-hidden weights
    w^(o) in R^J         : hidden-to-output weights
    b^(h) in R^J         : hidden-layer biases
    b^(o) in R           : output bias
    d = pJ + J + J + 1
"""
import torch
import torch.nn as nn
import copy


class ERNN(nn.Module):
    """Single-hidden-layer feedforward neural network for expectile regression."""

    def __init__(self, p: int, J: int, activation: str = "tanh"):
        super().__init__()
        self.p = p
        self.J = J

        # Hidden layer: input -> hidden  (p inputs, J hidden nodes)
        self.hidden = nn.Linear(p, J)
        # Output layer: hidden -> output  (J hidden, 1 output)
        self.output = nn.Linear(J, 1)

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "relu":
            self.activation = torch.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    @property
    def d(self):
        """Total parameter count: pJ + J + J + 1."""
        return self.p * self.J + self.J + self.J + 1

    def forward(self, x):
        """
        Forward pass.
        x: (N, p) tensor
        returns: (N,) tensor
        """
        h = self.activation(self.hidden(x))   # (N, J)
        out = self.output(h)                   # (N, 1)
        return out.squeeze(-1)                 # (N,)


def get_flat_params(model: ERNN) -> torch.Tensor:
    """
    Flatten model parameters in paper ordering:
        [vec(W^(h)), w^(o), b^(h), b^(o)]

    Note: nn.Linear stores weight as (out_features, in_features),
    so hidden.weight is (J, p). The paper defines W^(h) as (p, J),
    so vec(W^(h)) = hidden.weight.T.reshape(-1).
    """
    W_h = model.hidden.weight.data.T.reshape(-1)    # (p*J,) - input-to-hidden
    w_o = model.output.weight.data.reshape(-1)       # (J,)   - hidden-to-output
    b_h = model.hidden.bias.data.clone()             # (J,)   - hidden biases
    b_o = model.output.bias.data.clone()             # (1,)   - output bias
    return torch.cat([W_h, w_o, b_h, b_o])


def flatten_params_with_grad(model: ERNN) -> torch.Tensor:
    """
    Flatten model parameters in paper ordering [vec(W^(h)), w^(o), b^(h), b^(o)]
    WHILE PRESERVING the autograd graph.

    Unlike get_flat_params, which reads ``.data`` and therefore returns a tensor
    detached from the computation graph, this helper concatenates the live
    parameter tensors. It must be used whenever the flat parameter vector enters
    a differentiable objective (e.g. the linear correction term <c, theta> in the
    DPS-/CSL-ERNN surrogate); using the detached get_flat_params there would make
    the correction contribute a zero gradient, silently dropping it from the
    optimized objective.

    The ordering matches get_flat_params and compute_flat_grad exactly, so the
    dot product with a correction vector produced by compute_flat_grad is aligned
    component-by-component.
    """
    W_h = model.hidden.weight.T.reshape(-1)    # (p*J,) - input-to-hidden, grad-connected
    w_o = model.output.weight.reshape(-1)       # (J,)   - hidden-to-output
    b_h = model.hidden.bias                      # (J,)   - hidden biases
    b_o = model.output.bias                      # (1,)   - output bias
    return torch.cat([W_h, w_o, b_h, b_o])


def set_flat_params(model: ERNN, flat: torch.Tensor):
    """Set model parameters from flat vector in paper ordering."""
    p, J = model.p, model.J
    idx = 0

    # W^(h): input-to-hidden weights, stored as (p, J) in paper
    W_h = flat[idx: idx + p * J].reshape(p, J)
    model.hidden.weight.data = W_h.T.clone()       # nn.Linear stores (J, p)
    idx += p * J

    # w^(o): hidden-to-output weights
    w_o = flat[idx: idx + J]
    model.output.weight.data = w_o.reshape(1, J).clone()
    idx += J

    # b^(h): hidden biases
    b_h = flat[idx: idx + J]
    model.hidden.bias.data = b_h.clone()
    idx += J

    # b^(o): output bias
    b_o = flat[idx: idx + 1]
    model.output.bias.data = b_o.clone()


def compute_flat_grad(loss: torch.Tensor, model: ERNN) -> torch.Tensor:
    """
    Compute gradient of loss w.r.t. model parameters, returned in paper ordering.
    Does NOT call loss.backward() — uses torch.autograd.grad instead to avoid
    accumulating gradients.
    """
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=False,
                                allow_unused=True)

    # params order from nn.Module: hidden.weight, hidden.bias, output.weight, output.bias
    # hidden.weight grad: (J, p) -> transpose to (p, J) -> flatten
    # allow_unused=True means some grads may be None (e.g., penalty only touches hidden weights)
    dev = params[0].device
    g_W_h = grads[0].T.reshape(-1) if grads[0] is not None else torch.zeros(model.p * model.J, device=dev)
    g_b_h = grads[1].clone() if grads[1] is not None else torch.zeros(model.J, device=dev)
    g_w_o = grads[2].reshape(-1) if grads[2] is not None else torch.zeros(model.J, device=dev)
    g_b_o = grads[3].clone() if grads[3] is not None else torch.zeros(1, device=dev)

    # Paper ordering: [W^(h), w^(o), b^(h), b^(o)]
    return torch.cat([g_W_h, g_w_o, g_b_h, g_b_o])


def copy_model(model: ERNN) -> ERNN:
    """Deep copy of an ERNN model."""
    return copy.deepcopy(model)


def init_model(p: int, J: int, seed: int, activation: str = "tanh") -> ERNN:
    """Initialize an ERNN model with a fixed seed for reproducibility."""
    torch.manual_seed(seed)
    model = ERNN(p, J, activation)
    # Xavier initialization for better training stability
    nn.init.xavier_uniform_(model.hidden.weight)
    nn.init.zeros_(model.hidden.bias)
    nn.init.xavier_uniform_(model.output.weight)
    nn.init.zeros_(model.output.bias)
    return model
