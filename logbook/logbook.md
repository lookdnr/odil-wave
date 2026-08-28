# Logbook

## 27/05/2026: Weekly meeting summary

The next meeting will be on the 8th June.

Before the next meeting, the aim is for each of us to develop our own code (working collaboratively) to reproduce Fig 1 of the ODIL paper. The focus of this figure is comparing the solution of the 1D wave equation using a PINN and ODIL. It solves solely the forward problem. It also includes an ablation study which compares the error, training epochs, and execution time for both solutions. Some key details include:

- ODIL solution was found on a 25 x 25 grid.
- PINN consisted of two hidden layers with 25 neurons each.
-  For a given number of parameters N, ODIL represents the solution on a $\sqrt{N} \times \sqrt{N}$ grid, while PINN consists of two equally sized hidden layers with tanh activation.
- Collocation points are specific locations in the domain of a differential equation where the approximate solution is required to satisfy the equation exactly.
-  The number of collocation points for PINN is fixed and amounts to 8192 points inside the domain and 768 points for the initial and boundary conditions functions.
- The execution time is a product of the number of optimization epochs required to reach a 150% of the error obtained after 80000 epochs and an average execution time over the last 100 epochs.
- The ablation study featured a comparison of training with L-BFGS-B and Newton's method. The error metric was the $L2$ norm.
- In the figure, the plots compare against a reference solution computed using finite differences.

General order of affairs:

- Obtain reference solution
- Implement ODIL
- Implement PINN
- Experiment

Other notes fromt today:

- The end goal of this exercise is to understand the ODIl methodology in practice. Experiements should explore its limitations, et cetera. This can also be a guide for estimating computational requirements and time to solution.
- The hope is that ODIL will bypass the requirement for AWI.
- You can spawn a notebook on the RCS using [JupyterHub](https://jupyter.cx3.rcs.ic.ac.uk/hub/login?next=%2Fhub%2Fspawn).
- Consider setting up Weights & Biases to track experiments over time

## 28/05: progress update

Regarding the above task:

- Created `notebooks/reproduction/{reference.ipynb,odil.ipynb,pinn.ipynb}`
- Recreated the reference solution. Upon looking at the ODIL codebase, realised the initial condition the paper states in inconsistent with the plot. Mirrored their implementation as a result.
- Recreated the ODIL result. 
    - Discretised the wave equation as specified in the paper along with boundary conditions
    - Split functionality into a `Wavefield` class and a `DiscretePDELoss` class. The later has a `__call__` method which is used by `scipy.optimize` to evaluate the full functional
    - Implemented residual tracking and corresponding plotting functionality.
    - Ran, timed, and plotted results for ODIL for L-BFGS-B and Newton-CG with wavefield initialised with zeros and random values.

Next steps: 
- [ ] Implement the PINN approach
- [ ] Combine with reference and ODIL results in single plot including trace samples
- [ ] Investigate limitations of ODIL along with proper runtime statistics

## 29/05: progress update

I have implemented the PINN approach, but I am struggling to match their results. A key issue is the network is failing to replicate the high frequency information contained in the reference solution. I originally trained with just Adam and it produced a decent result but stagnated after ~10000 epochs. Now I am playing around with switching to L-BFGS later in the training to try to avoid getting stuck in local minima. I fear the network they said they used just doesn't have enough capacity to represent the high frequency data, though. The mismatch between what they stated and what they implemented in the code base for the reference solution is making me wary. They didn't mention anything about training strategy, and the architecture is slightly hazy, hence the difficulty. Will perservere.

## 01/06: Weekly meeting

Brief discussion of progress on the initial task. I have now got the ODIL and PINN solution working.

Also discussed how we could specialise. They suggested 3 options:

1. Software/ code optimisation focus: generic 2D/ 3D ODIL solver for forward problems in wave modelling
2. Domain focus: ODIL for generating accurate skull models as a preconditioner for FWI
3. Inverse focus: 2D FWI using ODIL

I also brought up the possibility of an UQ study, and they seemed interested, so maybe worth reading about. If that does not go ahead, opt. 1 and 3 seem interesting to me, but opt. 1 is probably less likely due to the need for GPU programming, etc.

## 02/06: progress update

I have finished the baseline ODIL and PINN figure reproduction. Next, I will either:

- Complete a similar ablation study
- Extend the ODIL framework to invert for a non-constant wavespeed

Note that the latter is effectively 1D FWI. In practice, we will need to add a data residual to the functional that measures the difference between the target and observed data at discrete point(s).

## 08/06: Weekly meeting

Today, we presented our results from the prior 2 weeks of experimentation. Over this period we have managed to accurately reproduce Figure 1 of the ODIL paper, experiment with algorithm ideas such as optimiser type, and implement baseline 1D and 2D implementations of FWI under ODIL. The presentation went well, and the supervisors seem satisfied with our progress so far.

We also discussed how we would split into individual avenues for investigation in the project. I think I am drawn between 

- Option 1: an efficient forward wave equation solver using the ODIL framework
- Option 2: a 2D FWI library using the ODIL framework

Option 1 is far more implementation focussed and will require a significant degree of software development, whereas option 2 is more specialised and will require a lot more experimentation around weighting, et cetera. Both are equally impactful.

This week I am focussing on writing up my project plan, starting a literature review, and drafting implementation ideas.

## 15/06: Weekly meeting

This is the first of the weekly meetings following our splitting into our separate avenues for the project. We began with a discussion of how we will collaborate. Specifically:

- We should work together to build a solid baseline
- We should consider the basic test cases we should implement. These will be shared.
    - Next two days: decide on spacings, models, geometries, and reference solutions (probably start with homogenous models with analytic solutions)
    - Plotting tools e.g., colour schemes should be unanimous
    - Define maximum problem sizes that are feasible to run locally
    - Source frequency vs target size
- Reference solutions should be comuted with stride
    - Forward modelling at a very fine grid
    - FWI for water start and perfect start
    - Probably use the Shepp-Logan phantom model

The others will rely on my forward solver. The others will probably tell me things to implement which I should follow. They should assume my main branch does what they want and pull from it. My work is the common point for each project.

Ben suggest that development should focus on modularity in the context of what we want to experiment with, for now. Our priority should be to be able to play around easily with models, grids, et cetera.

### Quick note on implementing sparse operators

Consider the 2D wave equation,

$$u_{tt} = c^2(u_{xx} + u_{yy}) = c^2 \nabla^2 u.$$

Now let

$$\nabla^2 = \delta_{xx} + \delta_{yy}.$$

We must find some way to represent this as a matrix operation.

Consider a simple central difference

$$\nabla^2 u \approx \frac {u_{i+1, j} -2 u_{i, j} + u_{i-1, j}}{\Delta x^2} + \frac {u_{i, j+i-1} -2 u_{i, j} + u_{i, j-1}}{\Delta y^2} := D_x +D_y.$$

We can represent $D_x$ as

$$\alpha\begin{bmatrix}
0 & 1 & 0\\
1 & -2 & 1\\
0 & 1 & 0\\
\end{bmatrix},$$

where $\alpha = \frac 1 {\Delta x^2}$. In this case then $D_y$ is simply

$$\frac \beta \alpha D_x,$$

where $\beta = \frac 1 {\Delta y^2}$. To turn this into the Laplacian operator, we need

$$
\begin{align*}
&\delta_{xx} = D_x \otimes I_{N_y},\\
&\delta_{yy} = D_y \otimes I_{N_x},\\
\Rightarrow\ &\nabla^2 = \delta_{xx} + \delta_{yy}
\end{align*}.$$

The core components we will intially implement are

- Optimiser
- DiscreteLoss
- DiscreteOperator
- Model
- Grid
- AcqusitionGeometry
- Domain

I am largely handling the optimisers, operators (w Milica), and losses.

## 17/06: Progress update

We have done quite a lot in the last few days, and spread the initial development between us. This should be recorded in the commit history, but this serves as a recognition of their contribution either way. Here is a breakdown

- Anton: Grid, AcqusitionGeometry, and VelocityModel classes
- Melica: various classes relating to conditions and stencil application
- Me: Optimiser classes, discrete loss classes, Wavefield class, SparseOperator classes 

There was a little bit of implementation friction, namely becase my package now has to contain two pathways, one for the forward problem (my focus) and one for the inverse problem (their focus). This requires the use of different optimisers, losses, stencils, et cetera, and is not quite as simple as "just stick the wavespeed model in and optimise". 

## 22/06: Weekly meeting

In this meeting, we walked through our progress from the past week. I informed the supervisors that I would be switching direction. Last week, I introduced a lot of machinery for gradient based optimisation and the inverse problem which I would never use. The scope of my repo blew up beyond that of my project, so I proposed a change:

- Use sparse matrix operators
- Optimise the loss (single step) via a direct solve

This is quite a significant reframe from the past week and will require quite a lot of refactoring which I have started now. This methodology will be much more efficient, and offer some interesting avenues.

We made good progress last week, and hopefully by the end of this week the scope will be well defined, and the package will be tested and more efficient for my use case. More updates to come.

# 24/06: Progress update

Over the last 3 days, I have performed a major restructure and validation of the core components of the library:

- Operators employ sparse matrices
- WaveEquation has a .matvec function which applies the above without ever forming the global operator matrix explicitly. For a 40x40 problem, this contributes a ~25% reduction in memory.
- Unit tests have been written that verify the Operators in the following manner:
    - Smoke tests for all operators
    - Tests for constant (1st derivative) and linear (2nd derivative) fields
    - Tests for polynomial exactness
    - Convergence tests for operators of all orders
    - Method of Manufactured solution for full wave equation
    - IC and BC enforcement checks

There are more tests that can be written, but it opens the door for us to proceed with the optimisation.

### Note on the MMS test

The wave equation operator is validated via a method of manufactured solution test.

I set

$$u_{\text{exact}} = t^2(x^2 + y^2),$$

such that

$$u_{tt} = c^2(u_{xx} + u_{yy}) - f \Leftrightarrow 2(x^2 + y^2) = 4c^2t^2 - f,$$

and 

$$f =  2(x^2 + y^2) - 4c^2 t^2.$$

Then I compute $Au$ for the given $u_{\text{exact}}$ and check that $Au \approx f$ as above. This validates that the operator does what I say it does.

### Note on project direction

In the last few days, I have realised that solving the forward problem using Gauss-Newton is actually just equivalent to solving the forward problem normally. The authors of the original paper basically say the same thing, which is nice, but it doesn't really offer anything novel at all. Thus, I should look at things like

- iterative optimisation
- non-linear forms that might emerge in other wave equations
- multi grid methods

## Progress update: 25/06

Right, so I have done some maths and (I think) proved that solving the GN forward problem is functionally equivalent to solving Au = f, which is not novel at all. It goes as follows:

The wave equation can be written

$$Au = f,$$

where $A$ is a matrix operator, $u$ is the amplitude vector, and $f$ is the source term vector.

Under ODIL, we seek to minimise

$$L(u) = \|r(u)\|^2,$$

where $r(u) = Au - f$ is teh residual vector. We can minimise this using a Gauss-Newton method.

GN approximates

$$r(u + \delta u) = r_k - J_k \delta u,$$

by a Taylor expansion. Here, $J_k$ is the Jacobian, and $\delta u$ is a vector point such that the residual is approximated at $u + \delta u$. Note however that in our (linear) case, the Jacobian is

$$\frac{\partial}{\partial u} r(u) = A,$$

the matrix operator. We will come back to this later.

Under this formulation, we can rewrite the loss as

$$L(u) = \|r_k - J_k \delta u\|^2.$$

Expanding this using the fact that $\|x\|^2 = x^Tx$, we get

$$L(u) = (r_k - J_k \delta u)^T(r_k - J_k \delta u).$$

This is a quadratic problem in $\delta u$.

Now, we want 

$$\nabla_{\delta u} L = J^T_k (r_k + J_k\delta u) = 0.$$

Rearranging, we arrive at

$$J^TJ \delta u = - J^T r,$$

which is the Gauss Newton normal equation. 

We can cancel the $J^T$ terms by multiplying both sides by $(J^T)^{-1}$:

$$\underbrace{(J^T)^{-1}J^T}_{= I} \ J \delta u = -\underbrace{(J^T)^{-1}J^T}_{= I}\ r$$

$$\Rightarrow J \delta u = -r.$$

But recall that $J \equiv A$ is constant, so $\delta u$ must be the exact minimiser of the loss. Sounds good so far, but if we realise that the residual for the current iterate $u_0$ is

$$r_0 = Au_0 - f,$$

and that 
$$\delta u = u_1 - u_0$$

then this becomes

$$J \delta u = - r \Rightarrow A(u_1 - u_0) = -(Au_0 - f),$$

which simplifies to 

$$Au_1 = f$$

thus, after a single GN step the updated iterate solves $Au = f$ exactly, so this approach is functionally equivalent to solving the forward problem directly. 

The issue with this is it is not novel at all, and is just a less efficient way around solving the problem directly. My direction needs to change. 

ODIL is specifically designed to solve inverse or non-linear problems. Here are some ideas:

- WRI
- Non-linear physics

More on this later :-\

## 29/06: Weekly meeting

Again in this meeting we all presented our progress from the past week. This will probably be the last joint meeting before we break off into one on one sessions with supervisors to boost productivity. There is not much to note from this one, I have a clear direction now.

I had a meeting with Carlos, and I explained the above issue I encountered. It turns out that this is not really an issue at all. In essence, there are many ways to solve a linear system, and proving what I have validates the ODIL approach: it is a new way of approximating a solution to the forward wave equation.

What has become clear in the past few days is that the whole task is not the outer optimisation loop at all. For a linear system, GN should trivially converge in a single step. The problem is that the conditioning number of the normal equations system is massive, so the inner solve crawls and GN does not converge.

In the ODIL paper, they mainly explore parabolic systems. However, the 2D wave equation is hyperbolic. The ODIL paper suggests that multigrid methods can accelerate convergence signfiicantly. However, MG methods are known to struggle for hyperbolic equations, since they tend to produce large, highly nonsymmetric systems [see here](https://excalibur-neptune.github.io/Documents/_static/TN-03_AReviewTimeSteppingTechniquesPreconditioningHyperbolicAnisotropicEllipticProblem.pdf). Thus, MG methods are not suitable for my problem, and the task remains to find a robsut preconditioner that allows for quick convergence for large systems.

The circumstance warrants a two legged approach:

- For small/medium systems, assemble the full operator matrix and do a sparse LU decomposition on A.
- For larger systems, use a [ParaDiag](https://icms.ac.uk/wp-content/uploads/2025/06/Josh-Hope-Collins.pdf)/ [block circulant](https://onlinelibrary.wiley.com/doi/full/10.1002/nla.2386) approach. Multigrid can be an ambitious arm of comparison, and if it fails in comparison to some preconditioner then this is still a nice thing to write about.

The framing then becomes: multigrid is the standard accelerator for least squares PDE discretisations (ODIL, FOSLS), but it doesn't transfer cleanly to the hyperbolic case, here's a wave-tailored preconditioner that does, characterised against a direct-solve ground truth.

## 06/07: Weekly meeting

In this meeting we broke off and spoke with individual supervisors, myself with Lluis. I walked him through my progress for the last week:

- Proper second order Higdon BCs
- Lagrange extrapolated ghost nodes
- Excellent speedup using the alpha circuant preconditioner

I also set a few objectives for this week:

- a reference solution on the Shepp Logan phantom
- looking at Levinson recursion over LU deomp
- isolating the solve to a particular region
- analysing dispersion and disspation errors

I am also working to make the preconditioner more efficient. Namely by exploiting symmetry in the FFT

- the input vector u is real
- as a result, the coefficients come in complex conjugate pairs
- this means we can just factorise every other pair and take the conjugate of it on exit

## 20/07: weekly meeting

Note: I forgot to push the notes for this week. I also realise I had not written an entry for the week prior. We held that meeting on the 15th, and there was not much to report.

In the last week, I have taken a step back from code. I was feeling a little burnt out and blinkered, so I decided to direct my energy at something else. I wrote up a significant chunk of the methodology and bought myself a little breathing room for later down the line. I have also settled on a definitive research question. My objective is to

> Define the regimes where ODIL is useful by analysing its error, performance, and wave-equation specifics versus a time stepping finite difference method.

After working on ODIL for months, I have realised its power is somewhat limited. There just does not seem to be a situation where time stepping is not preferable. My contribution will be a thorough investigation into whether or not this is true, and if there are any regimes where ODIl proves to be more useful. I hope my paper will act as a guideline for future use cases of ODIL in solving the wave equation.

## 27/07: Weekly meeting

Today, I spoke with Ben. When we met previously, Ben suggested I take a step back and consider what my long term research objectives are. After some discussion and some refinement myself after the meeting, I have approximately this:

The defining question is

> Are there regimes where ODIL is a preferable over finite-difference based time stepping for solving the 2D wave equation?

To support this objective, we intend to address the following questions in particular

- RQ1: Are there regimes where ODIL has favourable accuracy properties?
- RQ2: Are there regimes where ODIL has favourable computational performance properties?
- RQ3: How does ODIL handle the nuances of the numerical modelling of waves

My goal really is to provide a guide for anyone who is considering using ODIL to solve the wave equation, and perhaps hyperbolic problems for generally. I will report my intentions to the supervisors in the next meeting and get some feedback which I will log here.

## 03/08: weekly meeting

Today, I met with Carlos. I talked to him about my proposed research direction, and showed him some of the plots I made. He is happy with what I intend to do. He gave some general advice on structuring my report, etc, and we just had a bit, really. It wasn't a particularly long meeting.

Now is probably a good time for another general update. I have gotten back into the swing of things again and have been making good progress. I am running some accuracy sweeps against Devito (comparing to an analytical solution obtained by convolving the Green's function with the Ricker wavelet), and am getting some good results. 

Here are some of the core changes I have made to the code:

- Moved the pre-cached factorisations to on-the-fly. Slightly worse performance, but very easy fix for the memory issue. Don't know why I didn't think of this before lol
- Pinned the BLAS threads to 1. SciPy uses BLAS under the hood for splu factorisations, and I ran some experiements to verify it was actually helping because I ran into some roadblocks when thinking about parallelisation. Turns out, mutithreaded BLAS actualyl hurts performance quite a lot, even at 2 threads. As a result, I hard coded it to 1 via the `threadpoolctl` package.
- Changed to Hicks' style Kaiser-windowed sinc interpolation. This matches exactly what Devito does and quite tests against the analytical solution verify this. This is also a nice validation point.
- Added a number of scripts for running experiements, including a .pbs for the aforementioned accuracy sweep. I am persisting these and maintaining them well for the sake of reproducibility.
- Added a Grid.from_ppw classmethod to build the grid based on a required number of ppw. This is useful, because the CFL safety factor is an arbitrary knob that isn't particularly informative.
- Changed the ghost node interpolation. This was a major win for the ABC enforcement, I found that the cubic cap was effectively zeroing the second derivative in the Higdon equations, so they weren't doing their job.

Those were the major changes as far as I can remember. A lot has been done in the last week or so. I also touched up some of the plots/ plotting utilities for consistency, and have been doing a lot of writing. I have started to break into the intro/background and the methodology is much more fleshed out now.

It'll be a long month ahead, but things are starting to take shape.

## 10/08: weekly meeting

I met fairly breifly with Ben today and spoke about my research direction and current state. I have pretty much gotten all the results I want to get for my accuracy study, and I am nearly closing out the performance study, to. One thing he pointed out was that I have been leading my performance with hardware parallelism, but I shoudl enter with algorithmic complexity, so that is what I am working on getting some measurements for now. 

My report is coming along pretty well, but it is not very put together at the minute. In particular I am not happy with my introduction structure just yet, and I need to integrate experiment details into the methodology more clearly. Results and discussion will follow soon. Can't wait to be done!

# 24/08: weekly meeting

I missed the last entry, whoops.

Today was our final meeting of the IRP. I walked through my report with Carlos and discussed some results. Not too much to report here.

It has been very enjoyable to work on this project for the last few months, and I have gotten some interesting results. The next few days are for cleaning up my report and code. This will probably be my last entry.

Bye bye :)
:wq