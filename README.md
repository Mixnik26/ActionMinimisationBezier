# Action minimisation using Bézier curves

In this project I (attempt) to solve the Euler-Lagrange equations by finding the path that minimises a given action from a Lagrangian using a variational method. The key idea here is a to approximate the solution as an $n$-th degree Bézier curve and choose control points that minimise the action along that curve. As $n$ increases, the Bézier curve (as far as I can tell) seems to converge to the true solution.

If you are unfamiliar with Bézier curves, they are smooth curves defined by a selection of control points $\mathbf{P}_0,\mathbf{P}_1...\mathbf{P}_n$ which are polynomial in their parameter $t\in[0,1]$ with degree $n$. An $n$ degree Bézier curve always has exactly $n+1$ control points, two of which are the start and end points. They can be defined recursively as a linear interpolation of two $n-1$ degree Bézier curves, but are more commonly written as polynomials in $t$. Bézier curves are common in computer graphics and if you've done any work on Photoshop or Adobe illustrator, you've likely (perhaps unknowingly) run into them. See [wikipedia](https://en.wikipedia.org/wiki/Bézier_curve) for more info, and see [my desmos graph](https://www.desmos.com/calculator/6mnlzwksff) for an interactive cubic Bézier curve. Also, these curves can be generalised to Bézier surfaces and (in theory) Bézier volumes, so this method of action minimisation could generalise easily to much harder problems.

There is a bit of a grey area as to what is considered a *better* approximation. Many problems care only of the image of the solution curve, whereas others care more about the value of the action. Empirically, I've found that a better value of the action usually goes hand-in-hand with a better image, though this could generally not be the case; the Bézier curve could have the almost same image as the solution curve's, but parametrise a region too fast or slow and thus get a drastically different action. I will use the error in the action $\delta S$ as a measure of how well the Bézier curve approximates the true solution, and assume that a better action error means a better image. This choice is mostly due to ease of computation, but is also motivated by the error analysis below, where $\delta S$ not only has dependence on the deviation from the true path but also its derivative.

# Method
The user declares the Lagrangian $L$ for the problem, along with which degree $n$ Bézier curve they wish to approximate the solution with. An initial guess of control points and, depending on the problem, the integration bounds $t_i$ and $t_f$ are also supplied.<br>
The program defines a function $f(\mathbf{P}\_0, ... \mathbf{P}\_{n+1})$ which computes the action along the Bézier curve defined by its input control points using standard numerical integration techniques (which may require $t_i$ and $t_f$). This function is then passed to a minimiser, which seeks the control points that minimise $f$, starting from the initial guess.<br>
<br>
Action computation and minimisation is handled by scipy's `integrate.quad` and `optimize.minimize`. I find that sometimes `optimize.minimize` struggles to reach a suitable tolerance for larger Bézier degree (roughly $n\sim 5$ for the Brachistochrone example), and `integrate.quad` struggles to reach a suitable tolerance for some boundary conditions. In some Lagrangians, it's also possible that `optimize.minimize` gets stuck in a local minima instead of the global minima, which can be avoided with good choice of initial guess. In spite of this, I seem to be getting roughly 4th order convergence (sometimes even 5th order convergence) in the error of the action, however this is purely empirical and I have yet to prove it generally. I eventually want to implement this into a general Euler-Lagrange solver that can also handle IVP physics problems, which is why I choose the Lagrangian as an argument instead of the action.

# Numerically evaluating the action along a Bézier curve
The action is given by (parametrised in t):

$$S = \int_{t_i}^{t_f} L\left[t, \mathbf{B}, \frac{d\mathbf{B}}{dt}\right]\ \text{d}t$$

To evaluate this numerically, we want to ideally have an integral over the Bézier curve's parameter, say $\lambda \in [0,1]$; so we assert the change of variables $t \rightarrow \lambda$.<br>
In general, we would have:

$$S = \int_{0}^{1} \frac{dt}{d\lambda}\ L\left[t(\lambda), \mathbf{B}(\lambda), \frac{d\lambda}{dt}\ \frac{d\mathbf{B}(\lambda)}{d\lambda} \right]\ \text{d}\lambda$$

This defines a special set of problems where if the lagrangian is independent of $t$, and is such that the $dt/d\lambda$ cancels with the $d\lambda/dt$, then:

$$S = \int_{t_i}^{t_f} L\left[t, \mathbf{B}, \frac{d\mathbf{B}}{dt}\right]\ \text{d}t = \int_{0}^{1} L\left[\mathbf{B}(\lambda), \frac{d\mathbf{B}(\lambda)}{d\lambda} \right]\ \text{d}\lambda$$

This is perfect for problems like the Brachistochrone problem and optics problems where $L \sim \sqrt{\dot{x}^2 + \dot{y}^2 + ...}$ , in which cases $t_i$ or $t_f$ are not necessarily known.<br>
Back to the general case, we can still choose how to transform $t \rightarrow \lambda$, the simplest case being a linear transformation:

$$t = t_{i} + (t_{f} - t_{i})\lambda =: t_{i} + \Delta t\ \lambda$$

In which case:

$$S = \int_{0}^{1} \Delta t\ L\left[t_i + \Delta t\ \lambda, \mathbf{B}(\lambda), \frac{1}{\Delta t}\ \frac{d\mathbf{B}(\lambda)}{d\lambda} \right]\ \text{d}\lambda$$

This equation inherently assumes knowledge of $t_{i}$ and $t_{f}$, which is not necessarily known in some problems, especially in those where the goal is to determine $\Delta t$.

# Error analysis

The aim of this error analysis is to find the dependence of the error of the action $\delta S$ on the degree of the Bézier curve $n$.

It is straightforward to show that if the Bézier curve $\mathbf{B}(t) = [b_{1}(t), b_{2}(t), ...]^{T}$ deviates from the true solution $\mathbf{F}(t) = [f_{1}(t), f_{2}(t), ...]^{T}$ by some small deviation $\mathbf{\delta}(t) = [\delta_{1}(t), \delta_{2}(t), ...]^{T}$, then the error in the action $\delta S$ will be roughly bounded by an integral second order in $\mathbf{\delta}(t)$. To be more specific, for a Lagrangian $L[t, \mathbf{x}, \dot{\mathbf{x}}]$:

$$S(\mathbf{B}) = \int L[t, \mathbf{B}, \dot{\mathbf{B}}] \text{d}t$$

$$\quad = S(\mathbf{F}) + \int \sum_{i} \left[ \delta_i \left( \frac{\partial L}{\partial f_i} - \frac{d}{dt} \left[ \frac{\partial L}{\partial \dot{f_i}} \right] \right) \right] \text{d}t + \frac{1}{2} \int \sum_{i} \mathbf{\eta_i}^{T} H_{f} \mathbf{\eta_i}\ \text{d}t + \int O(\Vert\mathbf{\eta_i}\Vert^3) \text{d}t$$

$$\implies \delta S = S(\mathbf{B}) - S(\mathbf{F}) \approx \frac{1}{2} \int \sum_{i} \mathbf{\eta_i}^{T} H_{f} \mathbf{\eta_i}\ \text{d}t$$

where the Hessian matrix is defined as $`H_{f} = \begin{bmatrix} \partial^2 L/\partial f^2 & \partial^2 L/\partial f \partial \dot{f} \\\ \partial^2 L/\partial f \partial \dot{f} & \partial^2 L/\partial \dot{f}^2 \end{bmatrix}`$,<br> and $\mathbf{\eta_i} = [\delta_i, \dot{\delta_i}]^{T}$.

Applying the triangle inequality for integrals approximately bounds the error in the action to:

$$ \delta S < \frac{1}{2} \int \sum_{i} \Vert\mathbf{\eta_i}\Vert^2 \Vert H_{f}\Vert \text{d}t$$

As promised, $\delta S$ is roughly bounded by an integral second order in $\mathbf{\delta}(t)$, however there is also second order dependence on $\dot{\mathbf{\delta}}(t)$. This $\dot{\mathbf{\delta}}(t)$ dependence can be interpreted in this case as as a measure of how the variability of $\mathbf{F}(t)$ affects convergence. If the true solution turns sharply at a point, or generally if $\Vert \dot{\mathbf{F}}(t)\Vert$ is large, then a variational method such as this one would likely struggle to replicate it. For example, a Bézier curve would need quite a few control points to describe something like a loop in the path. This $\dot{\mathbf{\delta}}(t)$ dependence also shows that an approximation with almost the same image as the true solution can still be off by a significant amount by parametrising a region too fast or slow.

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

There is also a third polynomial form Bézier curve, but it is derived from the first definition so I will omit it. Using these definitions, in principle it is possible to find the dependence of $\Vert\mathbf{\eta_i}\Vert^2 \Vert H_{b}\Vert$ on $n$, however I have yet to do so.

I am also aware that $\beta_{i,n}$ are [Bernstein basis polynomials](https://en.wikipedia.org/wiki/Bernstein_polynomial) where it has been proven that: for a continuous function $f(x)$ on the interval $x\in[0,1]$, the [Bernstein polynomial](https://en.wikipedia.org/wiki/Bernstein_polynomial):

```math
B_n(f)(x) = \sum_{i=0}^n f(i/n)\ \beta_{i,n}(x)
```
satisfies

```math
\lim_{n\to\infty}B_{n}(f)(x) \to f(x)
```
uniformly on the interval $x \in [0,1]$. However, the term $f(i/n)$ indicates that the "control points" in this sum lie on $f(x)$, which is not necessarily the true for this use case. Although perhaps it is possible to prove that, for some cases, as $n \to \infty$, the control points $\mathbf{P}_i$ tend to lie on the true solution $\mathbf{F}(t)$ in which case this theorem could hold.