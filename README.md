# Controllability of RNN with constant inputs

This repository contains the code to implement the synthesis of constant inputs to drive a dynamical system (particularly a continous-time RNN) from an initial state to a target state in a given time horizon (Tamekue et al., 2025).

## Background

### Statement of the problem

Consider a nonlinear system with dynamics:

$$
\begin{align*}
\dot{x} & = N(x) + I	,\\
N(x) &= Wf(x) - Dx	,
\end{align*}
$$

where $x \in \mathbb{R}^{n}$ is the hidden state, $I \in \mathbb{R}^{n}$ is a constant input, $W \in \mathbb{R}^{n \times n}$ is the connectivity matrix, $f$ is an element-wise nonlinear activation function, $D \in \mathbb{R}^{n \times n}$ is a diagonal matrix with positive diagonal elements, indicating the decay rate.

Under very general assumptions, we can prove that the flow of the autonumous dynamics $\Phi_{t}: \mathbb{R}^{n} \to \mathbb{R}^{n}$ and the inverse flow $\Psi_{t}$ exists for all time $t$, and they are the inverse of each other:

$$
\begin{align*}
\partial_{t}\Phi_{t}(x^{0}) &= N(\Phi_{t}(x^{0}))	,\\
\partial_{t}\Psi_{t}(x^{0}) &= -N(\Psi_{t}(x^{0}))	,\\
\Psi_{t} &= \Phi_{t}^{-1}	.
\end{align*}
$$

**Goal:** for an initial state $x(0) = x^{0}$, a target state $x^{1} \in \mathbb{R}^{n}$, and a time horizon $T > 0$, we aim to find a constant control input $I \in \mathbb{R}^{n}$ which drives the system from $x(0)$ to a state $x(T)$ close to $x^{1}$ in time $T$. For linear systems, $x(T) = x^{1}$.

### Synthesis for linear systems

If $f$ is linear, without loss of generality, we assume $f$ is the identity function (otherwise multiply $W$ with the slope of $f$ and replace $f$ with the identity function) and define $A = W - D$. It can be proved that whenever $e^{TA} - \text{Id}$ is invertible, the following input $I$ drives the system from $x(0) = x^{0}$ to $x(T) = x^{1}$ in exactly time $T$:

$$
I = (e^{TA} - \text{Id})^{-1}A(x^{1} - e^{TA}x^{0}) = (e^{TA} - \text{Id})^{-1}A(x^{1} - x^{0}) - Ax^{0}
$$

where $\text{Id} \in \mathbb{R}^{n \times n}$ is the identity matrix.

Two sufficient conditions for the invertibility of $e^{TA} - \text{Id}$ are:

1. The spectral norm of the connectivity matrix $|||W|||$ is smaller than the minimum decay rate $\lambda_{\min}(D)$.
2. $\lambda = i\frac{2\pi l}{T}, \forall l \in \mathbb{Z}$ is not an eigenvalue of $A = W - D$, namely $e^{TA}$ does not have an eigenvalue of 1.

## Synthesis for nonlinear systems

For nonlinear systems, we can only provide an approximate to the actual input that solves the two-point boundary value problem. In the sections below, we present several methods to synthesize the input $I$ for nonlinear systems. We will denote the differential operator as $D$ which will appear in terms like $DN$ and $D\Phi$, and should not be confused with the diagonal decay matrix $D$ in the system dynamics.

### Linearization methods

#### `linearized`

As the name suggests, we let $A := DN(x^{0})$ and define the "linearized" system as:

$$
\begin{aligned}
\dot{z}(t) & = Az(t) + N(x^{0}) + I	, \\
z(0) & = z^{0} := \mathbf{0}	, \\
z^{1} & := x^{1} - x^{0}	.
\end{aligned}
$$

The following input $I$ will drive the linearized system from $z^{0}$ to $z^{1}$ in exact time $T$:

$$
I = (e^{TA} - \text{Id})^{-1}A(x^{1} - x^{0}) - N(x^{0})	.
$$

#### `linearzed_origin`

This method linearized the system at the origin (which is always a fixed point of the autonomous dynamics) rather than $x^{0}$. Let $A := DN(\mathbf{0})$, the synthesis is given by

$$
I = (e^{TA} - \text{Id})^{-1}A(x^{1} - e^{TA}x^{0})	.
$$

### Third generation methods

This section covers the methods presented in the final version of this paper.

#### `backward_nominal_state`

The synthesis is given by

$$
I = \left[ e^{-TDN(\Psi_{T}(x^{1}))} - \text{Id} \right]^{-1}DN(\Psi_{T}(x^{1}))\left( x^{0} - \Psi_{T}(x^{1}) \right)	.
$$

#### `forward_nominal_state`

The synthesis is given by

$$
I = \left[ e^{TDN(\Phi_{T}(x^{0}))} - \text{Id} \right]^{-1}DN(\Phi_{T}(x^{0}))\left( x^{1} - \Phi_{T}(x^{0}) \right)	.
$$

### Second generation methods

This section covers the methods developed during the review process of this paper.

#### `backward_initial_state`

The synthesis is given by

$$
I = \left[ \text{Id} - e^{-TDN(x^{0})} \right]^{-1}DN(x^{0})\left( \Psi_{T}(x^{1}) - x^{0} \right)	.
$$

#### `forward_final_state`

The synthesis is given by

$$
I = \left[ \text{Id} - e^{TDN(x^{1})} \right]^{-1}DN(x^{1})\left( \Phi_{T}(x^{0}) - x^{1} \right)	.
$$

### First generation methods

This section covers the methods proposed in the preprint version of this paper, which are based on the Jacobian of the backward or forward flow.

#### `backward_push`

This method is simply called "backward" method in the preprint. It is now called "backward_push" as it involves pushing forward a vector using the flow $\Phi_{T}$ evaluated at $\Psi_{T}(x^{1})$, the terminal state of the backward flow starting from $x^{1}$.

For nonlinear case, denote $D$ as the differential operator and define $B_{t}(x) := DN(\Psi_{t}(x)) = -D + Wf'(\Psi_{t}(x))$.

It can be proved that whenever $e^{TB_{T}(x^{1})} - \text{Id}$ is invertible, the following input $I$ drives the system from $x(0) = x^{0}$ to $x(T) = x^{1}$ in exact time $T$:

$$
\begin{aligned}
I & = [D \Psi_{T}(x^{1})]^{-1}[e^{TB_{T}(x^{1})} - \text{Id}]^{-1}B_{T}(x^{1})(\Psi_{T}(x^{1}) - x^{0}) + \zeta(T) \\
& = D \Phi_{T}(\Psi_{T}(x^{1}))[e^{TB_{T}(x^{1})} - \text{Id}]^{-1}B_{T}(x^{1})(\Psi_{T}(x^{1}) - x^{0}) + \zeta(T)
\end{aligned}
$$

where the error term $\zeta(T)$ is defined implicitly.

Two sufficient conditions for the invertibility of $e^{TB_{T}(x^{1})} - \text{Id}$ are:

1. $|f'(0)|\cdot|||W||| < \lambda_{\min}(D)$.
2. $\lambda = i\frac{2\pi l}{T}, \forall l \in \mathbb{Z}$ is not an eigenvalue of $B_{T}(x^{1})$.

Also note that if $f$ is the identity function, then $B_{T}(x) \equiv A = W - D$, and

$$
\begin{align*}
I &= e^{TA}(e^{TA} - \text{Id})^{-1}A(e^{-TA}x^{1} - x^{0})\\
&= (e^{TA} - \text{Id})^{-1}A(x^{1} - e^{TA}x^{0})
\end{align*}
$$

and the error term is zero.

#### `forward_pull`

This method is simply called "forward" method in the preprint. It is now called "forward pull" as it involves pulling back a vector using the backward flow $\Psi_{T}$ evaluated at $\Phi_{T}(x^{0})$, the terminal state of the forward flow starting from $x^{0}$.

The synthesis is given by 

$$
\begin{aligned}
I & = D \Psi_{T}(\Phi_{T}(x^{0}))[e^{-TA_{T}(x^{0})} - \text{Id}]^{-1}A_{T}(x^{0})(\Phi_{T}(x^{0}) - x^{1}) + \zeta(T)
\end{aligned}
$$

where the error term $\zeta(T)$ is defined implicitly. $A_{t}(x) := DN(\Phi_{t}(x))$.

## Implementation

### General considerations and computation of the Jacobian of the flow

We implement the synthesis using Pytorch and the `torchdiffeq` package. Here we use the "backward push" method as an example.

We need to compute three terms:

- The terminal state $\Psi_{T}(x^{1})$ of the inverse flow starting from $x^{1}$.
- The differential $DN$ at this terminal state.
- The inverse of the differential $D \Psi_{T}$ of the inverse flow starting from $x^{1}$, or equivalently, the differential $D \Phi_{T}$ starting from the terminal state.

The main difficulty is to compute the differential of the flow. We can do this in two ways:

1. Use `torch.autograd.functional.jvp` to compute the product of $D \Phi_{T}(\Psi_{T}(x^{1})) = [D\Psi_{T}(x^{1})]^{-1}$ and the vector $[e^{TB_{T}(x^{1})} - \text{Id}]^{-1}B_{T}(x^{1})(\Psi_{T}(x^{1}) - x^{0})$.
2. Use `torch.autograd.functional.jacobian` to get the jacobian, then compute everything as they appeared in the equation.

The first method is very fast and memory efficient. The second method requires computing the whole matrix, which is VERY SLOW AND MEMORY INTENSE (mostly because it is not vectorized), so we did not use it. 

**Special note on the compatibility of `torchdiffeq` and `torch.func`**

For `torchdiffeq` version `0.2.3` and `torch` version `2.0+`:

- `torchdiffeq` does not support `torch.func.vmap()`.
- Adaptive solvers in `torchdiffeq` does not support `torch.func.jvp()`.
- However, adaptive solvers do work well with `torch.autograd.functional.jvp()`.
- `torch.autograd.functional.jvp()` is not aware of batch dimensions. However, its results are still mathematically correct even when the input is batched.
  - More specifically, `torch.autograd.functional.jvp()` treats the input as if they were concatenated, and computes the jacobian of the multidimensional function with respect to the multidimensional input. However, the derivative of other batches with respect to any certain batch is zero anyway, so the result is still correct.

### Matrix inversion

In the "backward push" method, we need to compute the inverse of $e^{TB_{T}(x^{1})} - \text{Id}$; in the "forward pull" method, we need to compute the inverse of $e^{-TA_{T}(x^{0})} - \text{Id}$; in linearized methods, we need to compute the inverse of $e^{TA} - \text{Id}$. When $T$ gets large, these matrices can become very ill-conditioned, leading to numerical instability. Note that, however, we have the following identities:

$$
\begin{aligned}
[e^{TA} - \text{Id}]^{-1} & = [e^{TA}(\text{Id} - e^{-TA})]^{-1} = [\text{Id} - e^{-TA}]^{-1}e^{-TA}\\
[e^{-TA} - \text{Id}]^{-1} & = [e^{-TA}(\text{Id} - e^{TA})]^{-1} = [\text{Id} - e^{TA}]^{-1}e^{TA}
\end{aligned}
$$

In our examples, the decay part is usually large relatively to the nonlinear part, so the eigenvalues of $A$ usually have negative real parts. This means that the matrix $e^{TA}$ is usually well-conditioned, while $e^{-TA}$ is ill-conditioned. Therefore, this trick is most useful when using "forward pull" method. 

For the "forward pull" method, we have:

$$
[e^{-TA_{T}(x^{0})} - \text{Id}]^{-1} = [e^{-TA_{T}(x^{0})}(\text{Id} - e^{TA_{T}(x^{0})})]^{-1} = [\text{Id} - e^{TA_{T}(x^{0})}]^{-1}e^{TA_{T}(x^{0})}	.
$$

So the numerical synthesis is given by

$$
\begin{aligned}
I & = D \Psi_{T}(\Phi_{T}(x^{0}))[e^{-TA_{T}(x^{0})} - \text{Id}]^{-1}A_{T}(x^{0})(\Phi_{T}(x^{0}) - x^{1}) \\
  & = D \Psi_{T}(\Phi_{T}(x^{0}))[\text{Id} - e^{TA_{T}(x^{0})}]^{-1}e^{TA_{T}(x^{0})}A_{T}(x^{0})(\Phi_{T}(x^{0}) - x^{1})	.
\end{aligned}
$$

### Error for the initial value problem

In the "backward push" method, we need to compute $D\Phi_{T}(\Psi_{T}(x^{1}))$, while in the "forward pull" method, we need to compute $D\Psi_{T}(\Phi_{T}(x^{0}))$. In both cases, this is done by numerically solving the initial value problem (IVP; i.e., simulating the flow backward or forward) and computing the Jacobian of the operations. Note that theorectically, the IVP should give $\Phi_{T}(\Psi_{T}(x^{1})) = x^{1}$ and $\Psi_{T}(\Phi_{T}(x^{0})) = x^{0}$, but numerically, this is not always the case.

In practice, we found that $\Phi_{T}(\Psi_{T}(x^{1}))$ is always very close to $x^{1}$ (the "backward push" method), for whatever time horizon `T` and whatever systems we tried (including chaotic RNNs with weight scaling factor `g > 1`), using the default ODE solver in `torchdiffeq.odeint`. On the contrary, $\Psi_{T}(\Phi_{T}(x^{0}))$ could sometimes go way off from $x^{0}$ (the "forward pull" method), especially for large time horizon (`T > 20`) and close-to-chaotic systems (e.g., RNNs with `g > 0.8`), even when using an extremely small absolute tolerance (`atol=1e-14, rtol=1e-13`).

Our hypothesis to this phenomenon is the following. Most systems we tried were dominated by the decay term, so the numerical error is tolerated when simulating forward but will magnify when simulating backward. As a result, the "backward push" method is much more stable than the "forward pull" method for large $T$.

**Practical guideline:**

- Always check the third output of the `synthesize` function, which is the norm of the IVP error ($\|\Phi_{T}(\Psi_{T}(x^{1})) - x^{1}\|$ for the "backward push" method, and $\|\Psi_{T}(\Phi_{T}(x^{0})) - x^{0}\|$ for the "forward pull" method).
- If using the "backward push" method, usually the default setting of `odeint` should be enough.
- If using the "forward pull" method, you need to specify the following parameters for `odeint_kwargs` input of the `synthesize` function: `{'method': 'dopri8', 'rtol': 1e-13, 'atol': 1e-14}`.
  - For close-to-chaotic systems, it seems like the `radauIIA5` solver will be better, based on simulations with `scipy.integrate.solve_ivp`.
  - However, the current implementation of the `radauIIA5` solver in `torchdiffeq` is unacceptably slow. It seems like there is a [pull request](https://github.com/rtqichen/torchdiffeq/pull/263) in progress to fix this issue. We recommend trying out this solver for the "forward pull" method after the pull request is merged.

## Citation

If you use this code, please cite the following paper:

C. Tamekue, R. Chen and S. Ching, "On the Control of Recurrent Neural Networks using Constant Inputs," in *IEEE Transactions on Automatic Control*, doi: [10.1109/TAC.2025.3615934](https://doi.org/10.1109/TAC.2025.3615934).

```bibtex
@ARTICLE{11184738,
  author={Tamekue, Cyprien and Chen, Ruiqi and Ching, ShiNung},
  journal={IEEE Transactions on Automatic Control}, 
  title={On the Control of Recurrent Neural Networks using Constant Inputs}, 
  year={2025},
  volume={},
  number={},
  pages={1-16},
  keywords={Brain modeling;Brain;Biological system modeling;Recurrent neural networks;Control theory;Symmetric matrices;Nonlinear dynamical systems;Mathematical models;Jacobian matrices;Controllability;Control Synthesis;Nonlinear Control;Recurrent Neural Networks;Neurostimulation;Transcranial Direct Current Stimulation},
  doi={10.1109/TAC.2025.3615934}}

```

## References

Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems, 31. https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html

Schuessler, F., Dubreuil, A., Mastrogiuseppe, F., Ostojic, S., & Barak, O. (2020). Dynamics of random recurrent networks with correlated low-rank structure. Physical Review Research, 2(1), 013111. https://doi.org/10.1103/PhysRevResearch.2.013111
