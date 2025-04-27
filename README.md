# Action minimisation using Bezier curves

In this project I (attempt) to solve the Euler-Lagrange equations by finding the path that minimises a given action from a Lagrangian using a variational method. The key idea here is a to approximate the solution as an $n$-th degree Bezier curve and choose control points that minimise the action along that curve. As $n$ increases, the Bezier curve seems to converge to the true solution. I eventually want to implement this into a general Euler-Lagrange solver that can also handle IVP physics problems, which is why I choose the Lagrangian as an argument instead of the action.

Action computation and minimisation is handled by scipy's `integrate.quad` and `optimize.minimize`. I find that sometimes `optimize.minimize` struggles to reach a suitable tolerance for larger Bezier degree (roughly $n$~5 for the Brachistochrone example), and `integrate.quad` struggles to reach a suitable tolerance for some boundary conditions. In spite of this, I seem to be getting roughly 4th order convergence and sometimes even 5th order convergence, however this is purely empirical and I have yet to prove generality.

# Error analysis

It is straightforward to show that if the Bezier curve $\mathbf{B}(t) = [b_{1}(t), b_{2}(t), ...]^{T}$ deviates from the true solution $\mathbf{F}(t) = [f_{1}(t), f_{2}(t), ...]^{T}$ by some small deviation $\boldsymbol{\delta}(t) = [\delta_{1}(t), \delta_{2}(t), ...]^{T}$, then the error in the action $\delta S$ will be roughly second order in $\boldsymbol{\delta}(t)$. To be more specific, for a Lagrangian $L[t, \mathbf{x}, \dot{\mathbf{x}}]$:
```math
S(\textbf{B}) = \int L[t, \mathbf{B}, \dot{\mathbf{B}}] \text{d}t \\ \quad = S(\textbf{F}) + \int \sum_{i} \left[ \delta_i \left( \frac{\partial L}{\partial f_i} - \frac{d}{dt} \left[ \frac{\partial L}{\partial \dot{f_i}} \right] \right) \right] \text{d}t + \int \boldsymbol{\delta}^{T} H_{f} \boldsymbol{\delta}  + O(||\boldsymbol{\delta}||^{3})\text{d}t \\ \implies \delta S = S(\textbf{B}) - S(\textbf{F}) = \int \boldsymbol{\delta}^{T} H_{f} \boldsymbol{\delta}  + O(||\boldsymbol{\delta}||^{3})\text{d}t
```
where the Hessian matrix is defined as: $(H_{f})_{ij} = \frac{\partial^2 L}{\partial f_{i} \partial f_{j}}$ for entries $i,j$.

Since $\mathbf{B}(t) \approx \mathbf{F}(t)$, it is not unreasonable to think that the Hessian matrix in $\mathbf{B}(t)$ is close to that in $\mathbf{F}(t)$. Also applying the triangle inequality for integrals approximately bounds the error in the action to:
$$ \delta S < \int ||\boldsymbol{\delta}||^2 ||H_{b}|| \text{d}t$$