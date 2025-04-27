# Action minimisation using Bézier curves

In this project I (attempt) to solve the Euler-Lagrange equations by finding the path that minimises a given action from a Lagrangian using a variational method. The key idea here is a to approximate the solution as an $n$-th degree Bézier curve and choose control points that minimise the action along that curve. As $n$ increases, the Bézier curve (as far as I can tell) seems to converge to the true solution.

If you are unfamiliar with Bézier curves, they are smooth curves defined by a selection of control points $\mathbf{P}_0,\mathbf{P}_1...\mathbf{P}_n$ which are polynomial in their parameter $t\in[0,1]$ with degree $n$. An $n$ degree Bézier curve always has exactly $n+1$ control points. They can be defined recursively as a linear interpolation of two $n-1$ degree Bézier curves, but are more commonly written as polynomials in $t$. Bézier curves are common in computer graphics and if you've done any work on Photoshop or Adobe illustrator, you've likely (perhaps unknowingly) run into them. See [wikipedia](https://en.wikipedia.org/wiki/Bézier_curve) for more info, and see [my desmos graph](https://www.desmos.com/calculator/gqa1sxw0dj) for an interactive cubic Bézier curve.

Action computation and minimisation is handled by scipy's `integrate.quad` and `optimize.minimize`. I find that sometimes `optimize.minimize` struggles to reach a suitable tolerance for larger Bézier degree (roughly $n$~5 for the Brachistochrone example), and `integrate.quad` struggles to reach a suitable tolerance for some boundary conditions. In spite of this, I seem to be getting roughly 4th order convergence and sometimes even 5th order convergence, however this is purely empirical and I have yet to prove it generally. I eventually want to implement this into a general Euler-Lagrange solver that can also handle IVP physics problems, which is why I choose the Lagrangian as an argument instead of the action.

# Error analysis

It is straightforward to show that if the Bézier curve $\mathbf{B}(t) = [b_{1}(t), b_{2}(t), ...]^{T}$ deviates from the true solution $\mathbf{F}(t) = [f_{1}(t), f_{2}(t), ...]^{T}$ by some small deviation $\boldsymbol{\delta}(t) = [\delta_{1}(t), \delta_{2}(t), ...]^{T}$, then the error in the action $\delta S$ will be roughly bounded by an integral second order in $\boldsymbol{\delta}(t)$. To be more specific, for a Lagrangian $L[t, \mathbf{x}, \dot{\mathbf{x}}]$:

$$S(\mathbf{B}) = \int L[t, \mathbf{B}, \dot{\mathbf{B}}] \text{d}t$$

$$\quad = S(\mathbf{F}) + \int \sum_{i} \left[ \delta_i \left( \frac{\partial L}{\partial f_i} - \frac{d}{dt} \left[ \frac{\partial L}{\partial \dot{f_i}} \right] \right) \right] \text{d}t + \frac{1}{2} \int \sum_{i} \boldsymbol{\eta_i}^{T} H_{f} \boldsymbol{\eta_i}\ \text{d}t + \int O(\Vert\boldsymbol{\eta_i}\Vert^3) \text{d}t$$

$$\implies \delta S = S(\mathbf{B}) - S(\mathbf{F}) \approx \frac{1}{2} \int \sum_{i} \boldsymbol{\eta_i}^{T} H_{f} \boldsymbol{\eta_i}\ \text{d}t$$

where the Hessian matrix is defined as $`(H_{f})_{ij} = \partial^2 L/\partial f_{i} \partial f_{j}`$ for entries $i\ j$, and $\boldsymbol{\eta_i} = [\delta_i, \dot{\delta_i}]^{T}$.

Since $\mathbf{B}(t) \approx \mathbf{F}(t)$, it is not unreasonable to think that the Hessian matrix in $\mathbf{B}(t)$ might be close to that in $\mathbf{F}(t)$. Also applying the triangle inequality for integrals approximately bounds the error in the action to:

$$ \delta S < \frac{1}{2} \int \sum_{i} \Vert\boldsymbol{\eta_i}\Vert^2 \Vert H_{b}\Vert \text{d}t$$

As promised, $\delta S$ is roughly bounded by an integral second order in $\boldsymbol{\delta}(t)$, however there is also second order dependence on $\dot{\boldsymbol{\delta}}(t)$. This $\dot{\boldsymbol{\delta}}(t)$ dependence can be interpreted in this case as as a measure of how the variability of $\mathbf{F}(t)$ affects convergence. If the true solution is very sharp at a point, or generally if $\Vert \dot{\mathbf{F}}(t)\Vert$ is large, then a variational method such as this one would likely struggle to replicate it. For example, a Bézier curve would need quite a few control points to describe something like a loop in the path.

So far, we haven't defined $\mathbf{B}(t)$ to be a Bézier curve, so all the above holds for a general variation approach, but doing so now introduces dependence on the control points $\mathbf{P}_i$ and (more importantly) the degree $n$. There are multiple definitions of a Bézier curve as per [wikipedia](https://en.wikipedia.org/wiki/Bézier_curve):

```math
\mathbf{B}(t) = \sum_{i=0}^{n} \beta_{i,n}(t)\ \mathbf{P}_i, \quad 0\le t \le 1
```
```math
\text{with}\quad \beta_{i,n}(t) = \text{nCr}(n,i)\ (1-t)^{n-i}\ t^{i}
```

and, writing $\mathbf{B}_{P_0P_1...P_k}$ as the Bézier curve determined by any selection of points $P_0,P_1,...P_k$:
```math
\mathbf{B}(t) =  \mathbf{B}_{P_0P_1...P_n}(t) = (1-t)\mathbf{B}_{P_0P_1...P_{n-1}}(t) + t\mathbf{B}_{P_1P_2...P_n}(t), \quad 0\le t \le 1
```
```math
\text{with}\quad \mathbf{B}_{P_0}(t) = \mathbf{P}_0
```

There is also a third polynomial form Bézier curve, but it is derived from the first definition so I will omit it. Using these definitions, in principle it is possible to find the dependence of $\Vert\boldsymbol{\eta_i}\Vert^2 \Vert H_{b}\Vert$ on $n$, however I have yet to do so.

I am also aware that $\beta_{i,n}$ is a [Bernstein polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial) where it has been proven that: for a continuous function $f(x)$ on the interval $x\in[0,1]$, the Bernstein polynomial:

```math
B_n(f)(x) = \sum_{i=0}^n f(i/n)\ \beta_{i,n}(x)
```
satisfies

```math
\lim_{n\to\infty}B_{n}(f)(x) \to f(x)
```
uniformly on the interval $x \in [0,1]$. However, the term $f(i/n)$ indicates that the "control points" in this sum lie on $f(x)$, which is not necessarily the true for this use case. Although perhaps it is possible to prove that as $n \to \infty$, the control points $\mathbf{P}_i$ tend to lie on the true solution $\mathbf{F}(t)$ in which case this theorem could hold.