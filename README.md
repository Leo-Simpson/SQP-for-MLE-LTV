# Sequential Quadratic Programming for Maximum Likelihood Estimation of parameters in Linear Time Variant systems.
This repository contains code assiociated with the article [_"An Efficient Method for the Joint Estimation of System Parameters and
Noise Covariances for Linear Time-Variant Systems"_](https://arxiv.org/pdf/2211.12302.pdf)

## Prerequisites

Python packages

- numpy
- scipy
- matplotlib
- casadi
- cvxopt


# Description of the problem to solve

## Parametric Linear Time Variant Systems

We assume that the data is generated through the following dynamical system (linear, and with Gaussian Noise)
$$
\begin{align}
			x_{k+1} &= A(u_k; \alpha) x_k + b(u_k; \alpha) + w_k, && k = 0, \dots, N-1, \nonumber\\
			y_{k} &= C x_k + v_k, &&k = 0, \dots, N, \\
			w_k &\sim \mathcal{N}\left( 0, Q_k(\beta) \right), &&k = 0, \dots, N-1,  \nonumber \\
			v_k &\sim \mathcal{N}\left( 0, R_k(\beta) \right), &&k = 0, \dots, N,  \nonumber 
\end{align}
$$
where $\alpha$ are parameters of the nominal model, and $\beta$ are parameters corresponding to the perturbation model (Note: in the paper, this disctinction was not made).

Note that also the time-varying behavior comes from the inputs $u_k$, which can appear in the dynamics linearly or nonlinearly (Note: this is also different from the paper, where the concept of inputs was not present).

Finally, the matrix $C$ is fixed here (it would be possible to change that in the code without two many efforts however).

Also, the paremeters are assumed to be in some set defined with inequality constraints:
$$
\begin{align}
	\{ (\alpha, \beta) \in \mathbb{R}^{n_{\alpha}}\times \mathbb{R}^{n_{\beta}}  \; \big| \; h(\alpha, \beta) \geq 0 \},
\end{align}
$$
(Note that the inequality is opposite sign of how it is in the paper).

## The Estimation

We consider optimization problems for estimation of $\alpha$ and $\beta$.
These are basically maximizing the performance of a Kalman filter on the training data over the parameters $\alpha$ and $\beta$.


$$
\begin{align}
		&\underset{ \substack{
				\alpha, \beta, \bm{e}, \bm{S},
				%				 \bm{M},
				\bm{\hat{x}}, \bm{P}
			}
		}{\mathrm{minimize}} \; \sum_{k=0}^{N} e_k^\top \big(S_k\big)^{-1} e_k + \log \det S_k \nonumber \\
		& \mathrm{subject}  \, \mathrm{to} \, \nonumber
		\\&\phantom{ \mathrm{s} \,}
		S_k = C \, P_{k} \, C^{\top} + R(\beta), \nonumber
		%		\\&\phantom{ \mathrm{subject} \,}
		%		M_k \, S_k = I_{\ny},
		\\&\phantom{ \mathrm{s} \,}
		e_k = y_k - C \hat{x}_k,
		\\&\phantom{ \mathrm{s} \,}
		\hat{x}_{k+1} = A(u_k; \alpha)\big( \hat{x}_{k} + P_{k} \, C^{\top} S_k^{-1} e_k \big) + b(u_k; \alpha), \nonumber
		\\&\phantom{ \mathrm{s} \,}
		P_{k+1} = A(u_k; \alpha) \left(  P_k - P_k \, C^{\top} S^{-1} \, C \, P_k  \right) A(u_k; \alpha)^{\top} + Q(\beta) \nonumber,
        \\&\phantom{ \mathrm{s} \,}
        h(\alpha, \beta) \geq 0 \nonumber
		%		\phi_k(\alpha, P_k, S_k),
	\end{align}
$$

Regarding the cost function $L(\cdot, \cdot)$, two options are considered
$$
\begin{align}
		\begin{split}
			L_{\mathrm{MLE}}(e, S) & \equiv e^{\top} S^{-1} e + \log \det S, \\
			L_{\mathrm{PredErr}}(e, S) & \equiv \left\lVert e \right\rVert^2.
		\end{split}
	\end{align}
$$
The first of them is reffered as "MLE" because it corresponds to the Maximum-Likelihood problem, while the second is called "PredErr" because it corresponds to the Prediction Error Methods.

# Description of the algorithms

## IPOPT
One option is simply to call the solver IPOPT to solve the optimization problem (3) in this lifted form

## SQP

We propose a taylored SQP method.

It is composed by several steps:

- Propagate the state and covariances according to the Kalman filter equations to get $e_k$ and $S_k$.
- Computing the derivatives
    $\frac{\partial e_k}{\partial \alpha}$
    $\frac{\partial e_k}{\partial \beta}$
    $\frac{\partial S_k}{\partial \alpha}$
    $\frac{\partial S_k}{\partial \beta}$.

    This is done through "hand-made" forward AD.
- Computing gradient and Hessian approximation. 
   
    The gradient is always computed exactly. For the Hessian, we make some approximation.
    Regarding the "PredErr" method, we use Gauss-Newton Hessian approximation.
    Regarding "MLE" method, we make a similar one, which falls into the framework of Generalized Gauss-Netwon Hessian approximation after omitting the second-derivative of the term $\log \det S$ (indeed, this term is concave in $S$, while the other term is convex in $(e, S)$).
- Globalization via line-search. 
  
  Perform line-search in the direction found by backtracking until the Armijo condition is reached. 

# How to use the package

See the tutorial example in notebooks/pynotebooks/minimal_example.py

