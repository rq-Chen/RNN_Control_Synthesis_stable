"""visualizations.py - Functions for visualizing data."""

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from typing import Any, Sequence, Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# %% Visualize trajectories with PCA

def PlotTraj(dat: Tensor | ndarray, scatter_dat: Tensor | ndarray | None = None,
             PCspace: PCA | Tensor | ndarray | None = None, skip_steps: int = 0, ax=None,
             colormap: str = 'tab10', fixed_color: str | None = None, legend: bool = True, **kwargs):
    """Plot trajectories in the first 3 principal components

    Parameters:
    - dat: Tensor of shape (T, n_trials, n_features)
    - scatter_dat: Tensor of shape (n_trials, n_features) to scatter on the plot
    - PCspace: PCA object used to transform the data, or another dataset similar to `dat`.
    - skip_steps: Number of initial steps to skip in the plot (for clearer visualization)
    - ax: Matplotlib axis to plot on. If None, will create a new figure.
    - colormap: Colormap to use for coloring the trajectories.
    - fixed_color: If provided, will use this color for all trajectories.
    - legend: Whether to show the legend
    - kwargs: Additional arguments to pass to the plot function.
    
    If PCspace is provided, will plot data in the principal components of PCspace instead.
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    if skip_steps > 0:
        dat = dat[skip_steps:]
    
    if isinstance(dat, Tensor):
        dat = dat.cpu().numpy()
    T, n_trials, n = dat.shape

    if isinstance(scatter_dat, Tensor):
        scatter_dat = scatter_dat.cpu().numpy()
    
    if isinstance(PCspace, Tensor):
        PCspace = PCspace.cpu().numpy()
    if isinstance(PCspace, ndarray):
        assert PCspace.shape[-1] == n, "PCspace must have the same number of features as dat"
        PCspace = PCA(n_components=3).fit(PCspace.reshape(-1, n))
    else:
        PCspace = PCA(n_components=3).fit(dat.reshape(-1, n))
    
    dat_score = PCspace.transform(dat.reshape(-1, n)).reshape(T, -1, PCspace.n_components_)
    dat_score = dat_score.transpose(1, 0, 2)  # (n_trials, T, n_components)
    if fixed_color is not None:
        colors = [fixed_color] * n_trials
    else:
        colors = mpl.colormaps[colormap](np.linspace(0, 1, n_trials))

    if scatter_dat is not None:
        scatter_score = PCspace.transform(scatter_dat)  # (n_trials, n_components)

    for i in range(n_trials):
        p1 = ax.plot(dat_score[i, 0, 0], dat_score[i, 0, 1], dat_score[i, 0, 2], 'o', color=colors[i], **kwargs)
        p2 = ax.plot(dat_score[i, -1, 0], dat_score[i, -1, 1], dat_score[i, -1, 2], 'x', color=colors[i], **kwargs)
        p3 = ax.plot(dat_score[i, :, 0], dat_score[i, :, 1], dat_score[i, :, 2], color=colors[i], **kwargs)
        if scatter_dat is not None:
            p4 = ax.plot(scatter_score[i, 0], scatter_score[i, 1], scatter_score[i, 2], '*', color=colors[i], **kwargs)
        if i == 0 and legend:
            if scatter_dat is not None:
                ax.legend([p1[0], p2[0], p3[0], p4[0]], ['Start', 'End', 'Trajectory', 'Target'])
            else:
                ax.legend([p1[0], p2[0], p3[0]], ['Start', 'End', 'Trajectory'])
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    return ax


# %% Tests

if __name__ == "__main__":
    from torchdiffeq import odeint
    from Synthesis import get_RNN
    mdl = get_RNN(16, 2)
    x0 = torch.randn(10, 16)
    sims = odeint(mdl, x0, torch.linspace(0, 10, 100))
    PlotTraj(sims, skip_steps=5)
    plt.show()