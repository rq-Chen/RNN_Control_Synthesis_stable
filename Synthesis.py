"""Synthesis control input for RNNs.
"""

import torch
from torch import nn, Tensor
from torch.func import jacrev, vmap
from torch.autograd.functional import jacobian, jvp
from torchdiffeq import odeint
from typing import Any, Tuple
from scipy.io import loadmat
from warnings import warn

# Use double precision for all tensors
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)


# %% Models

class RNN(nn.Module):
    def __init__(self, n: int, act: nn.Module = torch.tanh, W: Tensor | None = None, D: Tensor | None = None):
        super().__init__()
        if W is None:
            W = torch.randn(n, n)
        if D is None:
            D = torch.randn(n)
        self.W = nn.Parameter(W, requires_grad=False)
        self.D = nn.Parameter(D.flatten(), requires_grad=False)
        self.act = act
        self.n = n
    def forward(self, t: Any, x: Tensor) -> Tensor:
        return self.act(x) @ self.W.T - self.D * x


class ControlledMdl(nn.Module):
    def __init__(self, mdl: nn.Module, I: Tensor):
        super().__init__()
        self.mdl = mdl
        self.I = nn.Parameter(I, requires_grad=False)
    def forward(self, t: Any, x: Tensor) -> Tensor:
        return self.mdl(t, x) + self.I


# %% Activation functions

# Piecewise linear activation function
class PiecewiseLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        return torch.clamp(x, -1, 1)

 
# MINDy activation function
class MINDyAct(nn.Module):
    """Activation function in Mesoscale Individualized NeuroDynamic model.

    See https://doi.org/10.1016/j.neuroimage.2020.117046 for details.
    """
    def __init__(self, alpha: Tensor | None = None, b: float | Tensor = 20/3):
        super().__init__()
        if alpha is None:
            alpha = torch.relu(torch.randn(100) * 0.1 + 5) + 0.01
        self.alpha = nn.Parameter(alpha.flatten(), requires_grad=False)
        self.A2 = nn.Parameter(self.alpha ** 2, requires_grad=False)
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
        self.n = alpha.numel()
    def forward(self, x):
        return (torch.sqrt(self.A2 + (self.b * x + 0.5) ** 2) -
                torch.sqrt(self.A2 + (self.b * x - 0.5) ** 2))


# %% Functions to generate models

def get_random_linear(N: int = 16, max_eig: float = -0.1, dvc: str|torch.device = 'cpu') -> RNN:
    """Generate a linear RNN with random dynamic matrix A.

    Args:
        N: Dimension of the state space.
        max_eig: Desired maximum of the real part of the eigenvalues of A.
        dvc: Device to put the models on.
    
    Returns:
        mdl: a linear RNN with random connectivity and uniform decay.
    
    Elements of W are drawn from a normal distribution with mean 0 and
    variance 1/N. The eigenspectrum of such matrix typically falls within
    the unit circle uniformly (with outliers). The decay is set to shift
    the maximum of the real part of the eigenvalues to max_eig.
    """
    W = torch.randn(N, N) / torch.sqrt(torch.Tensor([N]))
    D = torch.tile(torch.linalg.eigvals(W).real.max() - max_eig, (N,))
    mdl = RNN(N, nn.Identity(), W.to(dvc), D.to(dvc))
    return mdl


def get_RNN(N: int = 16, K: int = 1, g: float = 0.9, act: nn.Module = torch.tanh, dvc: str|torch.device = 'cpu',
            theta_i: list = []) -> RNN:
    """Generate an tanh RNN with random plus low-rank connectivity J + mn^T.

    Args:
        N: Dimension of the state space.
        K: Rank of the low-rank component of the connectivity.
        g: Scaling factor for the random connectivity.
        dvc: Device to put the models on.
        theta_i: expectation of n^{\\top}J^{i}m for i = 0, 1, ... of each low-rank
            component mn^{\\top}.
    
    Returns:
        mdl: an RNN with random plus low-rank connectivity J + P.
    
    See https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.013111
    for details. Generally speaking, the eigenspectrum of W lies uniformly within a
    circle centered at the origin with radius g. Besides, due to the existence of the
    low-rank component there will also be several real eigenvalues that could be outside
    the circle. If theta_i = 0 for all i > 0 (which is typically the case), the
    only one such eigenvalue has an expectation of theta_i[0]. If theta_i is not set,
    this eigenvalue will be small and within the unit circle.

    Remember that the Jacobian of the model at x = 0 is Wf'(0) - Id. For tanh, f'(0) = 1,
    so the eigenvalues of the Jacobian is that of W shifted by -1. Therefore, to make the
    origin unstable, set theta_i[0] to be larger than 1.
    """
    J = torch.randn(N, N) * (g / torch.sqrt(torch.Tensor([N])))
    W = J
    for k in range(K):
        m = torch.randn(N)
        if len(theta_i):
            n = torch.zeros_like(m)
            Jim = m
            for (i, theta) in enumerate(theta_i):
                n += theta / (g ** (2 * i)) * Jim
                Jim = J @ Jim
            n /= N
        else:
            n = torch.randn(N) / N
        W += torch.outer(m, n)
    D = torch.ones(N, dtype=default_dtype)
    mdl = RNN(N, act, W.to(dvc), D.to(dvc))
    return mdl


def get_all_MINDys(f: str = 'MINDy100.mat', n: int = 0, dvc: str|torch.device = 'cpu') -> list[RNN]:
    """load MINDy parameters from a mat file

    The models came from the original MINDy paper but for 100 instead of 400 nodes.
    The data file is a (53 subjects, 2 sessions) cell array, each containing a
    structure with fields 'W', 'alpha', 'D'. We will return a list of 106 RNNs.
    """
    allMdl = loadmat(f)['allMdl']
    if n > 0:
        allMdl = allMdl[:n]
    allMdl = allMdl.flatten()
    allMdl = [RNN(n = 100, act = MINDyAct(alpha = torch.tensor(mdl['alpha'][0, 0], dtype=default_dtype)),
                    W=torch.tensor(mdl['W'][0, 0], dtype=default_dtype),
                    D=torch.tensor(mdl['D'][0, 0], dtype=default_dtype)).to(dvc) for mdl in allMdl]
    return allMdl


# %% Utility functions

def expM_minus_Id_inv_times_v(M: Tensor, v: Tensor, smart: bool = True) -> Tuple[Tensor, Tensor]:
    """Compute (exp(M) - Id)^{-1}v.

    Args:
        M: Tensor, (..., n, n) where ... is the batch size and n is the size of the matrix.
        v: Tensor, (..., n) where ... is the batch size and n is the size of the vector.
        smart: bool. If False, compute (exp(M) - Id)^{-1}v directly. If True, see following.
       
    Returns:
        res: Tensor, (..., n), reuslt of (exp(M) - Id)^{-1}v.
        cond: Tensor, (...), condition number of the matrix that we ACTUALLY invert.
    
    This function computes (exp(M) - Id)^{-1}v in two different ways:
    - (exp(M) - Id)^{-1}v; or
    - (Id - exp(-M))^{-1}exp(-M)v
    and returns the one where the matrix to be inverted has smaller condition number.
    """
    N = M.shape[-1]
    M1 = torch.linalg.matrix_exp(M) - torch.eye(N, device=M.device, dtype=M.dtype)
    if not smart:
        res = torch.linalg.solve(M1, v)
        cond = torch.linalg.cond(M1)
        return res, cond
    expMinusM = torch.linalg.matrix_exp(-M)
    M2 = torch.eye(N, device=M.device, dtype=M.dtype) - expMinusM
    cond1 = torch.linalg.cond(M1)
    cond2 = torch.linalg.cond(M2)
    idx1 = cond1 < cond2
    idx2 = torch.logical_not(idx1)
    res = torch.empty_like(v)
    cond = torch.empty_like(cond1)
    if idx1.any():
        res[idx1] = torch.linalg.solve(M1[idx1], v[idx1])
        cond[idx1] = cond1[idx1]
    if idx2.any():
        vnew = torch.squeeze(expMinusM[idx2] @ torch.unsqueeze(v[idx2], -1), -1)
        res[idx2] = torch.linalg.solve(M2[idx2], vnew)
        cond[idx2] = cond2[idx2]
    return res, cond


# %% Input synthesis

def synthesize(mdl: nn.Module, x0: Tensor, x1: Tensor, T: float = 1.,
               method: str = "backward_nominal_state", odeint_kwargs: dict = {},
               smart_inverse: bool | None = None) -> Tuple[Tensor, Tensor, Tensor]:
    """Synthesize control input for RNNs.

    Args:
        mdl: RNN model.
        x0: Initial state, (..., n) where ... is the batch size and n is RNN size.
        x1: Target state, (..., n).
        T: Time horizon.
        method: See README.md for details. Also includes undocumented methods:
            "special": an alternative synthesis when |||W||| < Dmin.
            "naive": Set I = -mdl(None, x1).
        odeint_kwargs: dictionary, other odeint arguments.
        smart_inverse: if provided, pass it to expM_minus_Id_inv_times_v.

    Returns:
        I: Control input, (..., n).
        Mcond: Conditional number of the key matrix:
            "naive": None.
            others: shape (...,).
        ivp_error: error of the initial value problem:
            "backward_push": ||x1 - PhiTPsiTx1||, shape (...,). 
            "forward_pull": ||x0 - PsiTPhiTx0||, shape (...,). 
            others: None
    """
    if smart_inverse is None:
        if method in ["forward_pull", "backward_initial_state", "backward_nominal_state",
                      "linearized", "linearized_origin"]:
            smart_inverse = True
        else:
            smart_inverse = False

    dvc = x0.device
    N = x0.shape[-1]
    old_shape = x0.shape
    x0 = x0.reshape(-1, N)
    x1 = x1.reshape(-1, N)
    fullT = torch.tensor([0, T], dtype=default_dtype).to(dvc)
    revFullT = torch.tensor([T, 0], dtype=default_dtype).to(dvc)
    DN = vmap(jacrev(lambda x: mdl(None, x)))

    ## 3rd generation methods ##

    if method == "backward_nominal_state":
        PsiTx1 = odeint(mdl, x1, revFullT, **odeint_kwargs)[-1]
        DNPsiTx1 = DN(PsiTx1)
        v = torch.squeeze(DNPsiTx1 @ torch.unsqueeze(x0 - PsiTx1, -1), -1)
        I, Mcond = expM_minus_Id_inv_times_v(-T * DNPsiTx1, v, smart=smart_inverse)
        ivp_error = None

    elif method == "forward_nominal_state":
        PhiTx0 = odeint(mdl, x0, fullT, **odeint_kwargs)[-1]
        DNPhiTx0 = DN(PhiTx0)
        v = torch.squeeze(DNPhiTx0 @ torch.unsqueeze(x1 - PhiTx0, -1), -1)
        I, Mcond = expM_minus_Id_inv_times_v(T * DNPhiTx0, v, smart=smart_inverse)
        ivp_error = None

    ## 2nd generation methods ##

    elif method == "backward_initial_state":
        PsiTx1 = odeint(mdl, x1, revFullT, **odeint_kwargs)[-1]
        DNx0 = DN(x0)
        v = -torch.squeeze(DNx0 @ torch.unsqueeze(PsiTx1 - x0, -1), -1)  # Note the minus sign
        I, Mcond = expM_minus_Id_inv_times_v(-T * DNx0, v, smart=smart_inverse)
        ivp_error = None
    
    elif method == "forward_final_state":
        PhiTx0 = odeint(mdl, x0, fullT, **odeint_kwargs)[-1]
        DNx1 = DN(x1)
        v = -torch.squeeze(DNx1 @ torch.unsqueeze(PhiTx0 - x1, -1), -1)  # Note the minus sign
        I, Mcond = expM_minus_Id_inv_times_v(T * DNx1, v, smart=smart_inverse)
        ivp_error = None

    ## 1st generation methods ##

    elif method == "backward_push":
        PsiTx1 = odeint(mdl, x1, revFullT, **odeint_kwargs)[-1]
        BTx1 = DN(PsiTx1)
        v = torch.squeeze(BTx1 @ torch.unsqueeze(PsiTx1 - x0, -1), -1)
        tangent, Mcond = expM_minus_Id_inv_times_v(T * BTx1, v, smart=smart_inverse)

        PhiT = lambda x: odeint(mdl, x, fullT, **odeint_kwargs)[-1]
        x1_numeric, I = jvp(PhiT, (PsiTx1,), (tangent,))
        ivp_error = torch.linalg.norm(x1_numeric - x1, dim=-1)

    elif method == "forward_pull":
        PhiTx0 = odeint(mdl, x0, fullT, **odeint_kwargs)[-1]
        ATx0 = DN(PhiTx0)
        v = torch.squeeze(ATx0 @ torch.unsqueeze(PhiTx0 - x1, -1), -1)
        tangent, Mcond = expM_minus_Id_inv_times_v(-T * ATx0, v, smart=smart_inverse)

        PsiT = lambda x: odeint(mdl, x, revFullT, **odeint_kwargs)[-1]
        x0_numeric, I = jvp(PsiT, (PhiTx0,), (tangent,))
        ivp_error = torch.linalg.norm(x0_numeric - x0, dim=-1)
    
    ## Linearization and other methods ##

    elif method == "linearized":
        A = DN(x0)
        v = torch.squeeze(A @ torch.unsqueeze(x1 - x0, -1), -1)  # x1New = x1 - x0, x0New = 0
        I, Mcond = expM_minus_Id_inv_times_v(T * A, v, smart=smart_inverse)
        I -= mdl(None, x0)  # counteract the drift
        ivp_error = None

    elif method == "linearized_origin":
        A = DN(torch.zeros(1, N, device=dvc)).squeeze(0)
        eTA = torch.linalg.matrix_exp(T * A)
        v = (x1 - x0 @ eTA.T) @ A.T
        # Tile the matrix to the batch size
        I, Mcond = expM_minus_Id_inv_times_v(
            torch.tile(T * A, (x0.shape[0], 1, 1)), v, smart=smart_inverse)
        ivp_error = None

    elif method == "special":
        W = mdl.W
        D = mdl.D
        Dmin = D.min()
        if torch.linalg.matrix_norm(W, ord=2) >= Dmin:
            warn("The special method is only valid when |||W||| < Dmin.")
        PsiTx1 = odeint(mdl, x1, revFullT, **odeint_kwargs)[-1]
        BTx1 = DN(PsiTx1)
        DPsiTx1 = torch.stack([jacobian(lambda x: odeint(mdl, x, revFullT, **odeint_kwargs)[-1], x1v) for x1v in x1])
        M1 = torch.linalg.solve(BTx1, DPsiTx1)
        DNx0 = torch.stack([jacobian(lambda x: mdl(None, x), x0v) for x0v in x0])
        M0 = DNx0.inverse()
        M = M0 - M1
        v = PsiTx1 - x0
        Mcond = torch.linalg.cond(M)
        I = torch.linalg.solve(M, v)
        ivp_error = None
    
    elif method == "naive":
        I = -mdl(None, x1)
        Mcond, ivp_error = None, None

    else:
        raise ValueError("Invalid method.")
    
    # Reshape the output to match the input shape
    I = I.reshape(old_shape)
    if Mcond is not None:
        Mcond = Mcond.reshape(old_shape[:-1])
    if ivp_error is not None:
        ivp_error = ivp_error.reshape(old_shape[:-1])
    
    return I, Mcond, ivp_error
    

# %% Tests

if __name__ == "__main__":

    dvc = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # N = 100
    # mdls = get_all_MINDys(dvc=dvc)
    # mdl = mdls[torch.randint(len(mdls), (1,)).item()]
    # x0 = torch.randn(30, N, device=dvc)
    # x1 = torch.randn(30, N, device=dvc)
    # T = 32

    N = 100
    mdl = get_RNN(N, K=0, g=0.5, dvc=dvc)
    print("Spectral norm of W:", torch.linalg.matrix_norm(mdl.W, ord=2))
    x0 = torch.randn(30, N, device=dvc)
    x1 = torch.randn(30, N, device=dvc)
    T = 32

    # N = 100
    # mdl = get_random_linear(N, dvc=dvc)
    # x0 = torch.randn(30, N, device=dvc)
    # x1 = torch.randn(30, N, device=dvc)
    # T = 10

    method = "backward_nominal_state"
    odeint_kwargs = {'method': 'dopri8', 'rtol': 1e-13, 'atol': 1e-14}

    I, Mcond, ivp_error = synthesize(mdl, x0, x1, T, method=method, odeint_kwargs=odeint_kwargs)[:3]
    cMdl = ControlledMdl(mdl, I)
    xt = odeint(cMdl, x0, torch.linspace(0, T, 300, device=dvc), **odeint_kwargs)
    xdrift = odeint(mdl, x0, torch.linspace(0, T, 300, device=dvc), **odeint_kwargs)
    
    x0 = x0.cpu()
    x1 = x1.cpu()
    xt = xt.cpu()
    xdrift = xdrift.cpu()
    print("Method:", method)
    print("Conditional number of key matrix:")
    print(Mcond)
    print("Initial value problem error:")
    print(ivp_error)
    print("Norm of control input:")
    print(torch.linalg.norm(I, dim=-1))
    print("Endpoint error:")
    print(torch.linalg.norm(xt[-1] - x1, dim=-1))
    print("Norm of x1 - x0:")
    print(torch.linalg.norm(x1 - x0, dim=-1))
    
    from visualizations import PlotTraj
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})
    PlotTraj(xdrift, ax=ax[0])
    ax[0].set_title("Drift")
    PlotTraj(xt, x1, PCspace=xdrift, ax=ax[1])
    err = torch.linalg.norm(xt[-1] - x1, dim=-1) / torch.linalg.norm(x1 - x0, dim=-1)
    ax[1].set_title(f"Controlled, median relative endpoint error: {err.median():g}")
    fig.suptitle(f"Method: {method}, time: {T:.2f} units")
    plt.show()
    
    