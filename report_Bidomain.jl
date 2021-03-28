### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f0083160-8c97-11eb-04c0-ad918b035d75
begin 
	ENV["LANG"]="C"
	using Pkg
	Pkg.activate(mktempdir())
	using Revise
	Pkg.add("Revise")
	Pkg.add(["PyPlot","PlutoUI","ExtendableGrids","GridVisualize", "VoronoiFVM", "NLsolve"])
	using PlutoUI,PyPlot,ExtendableGrids,VoronoiFVM,GridVisualize,NLsolve
	PyPlot.svg(true)
end;

# ╔═╡ 0a475948-8c98-11eb-0787-2546680a44e6
md"""
# Bidomain Model
Author : Grégoire Pourtier (group 2, biology) \
Supervisor : Jürgen Fuhrmann
"""

# ╔═╡ d846fa24-8c98-11eb-3524-8d570cf24952
TableOfContents()

# ╔═╡ 496173c0-8c98-11eb-086e-b56945782ecf
md"""
## 1. Introduction : problem overview

The bidomain equations was first suggest by Otto Schmitt, an american enginner, in 1969 before being formulated mathematically at the end of 70s.
It model the propagation of electrical potential waves in the myocardium, also called heart muscle.

The bidomain equations are part of a long list of models that describe the phenomena of cardiac electrophysiology. It is the most complete mathematical model for describing the spread of cardiac electrical activity.
 The model is expressed through a nonlinear system of partial differential equations, composed of a coupled parabolic and elliptic partial differential equations and one ordinary differential equation (originally just a set of coupled partial differential equations).

The coupled partial differential equations desribed two electrical potential, $u_i$ which is the intracellular potential and $u_e$ which is the extracellular potential.
Then the ordinary differential equation models the ion activity, $v$, through the cardiac cell membranes. We refered to a coupled set of partial differential equations since the intracellular and extracellular potentials are solved simultaneously.
We chose to use the representation introduced in [1] for the bidomain model's equations. We have the following three species problem :


$\begin{align}

	\frac{\partial u}{\partial t} = \frac{1}{\epsilon} f(u ,v) +  \nabla\cdot (\sigma_{i} \nabla u) + \nabla\cdot (\sigma_{i} \nabla u_{e})\; \quad (1) \\


	\nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}) = 0\; \quad (2) \\


	\frac{\partial v}{\partial t} = \epsilon g(u, v)\; \quad (3)

\end{align}$

where

$u = u_i -u_e$

u is defined as the action or transmembrane potential, $\sigma_i$ and $\sigma_e$ are second order tensors representing, respectively, the intracellular and the extracellular tissue’s electrical conductivity in each spatial direction and $\epsilon$ is a parameter linked to the ratio between the repolarization rate and the tissue excitation rate.
Moreover, we decided to follow the model from [1] to describe the ionic activity across the cellular membrane, i.e. that we're using the FitzHugh– Nagumo equations, which are a simplification of the Hodgkin–Huxley equations, therefore we can write $f$ and $g$ as :

$\begin{align}
	f(u, v)= u−\frac{u^{3}}{3}−v \\
	g(u, v)= u + \beta - \gamma v \\
\end{align}$


\

If we investigate into the formulation of the problem above, we noticed that the system of partial differential equations models a nonlinear reaction-diffusion system.

By examining the system, we can see that the first equation of the system depends on time (since it's a parabolic PDE), there is two flux terms, which are linear and one nonlinear reaction term. The source term is equal to zero.
Then in the second PDE, there is only one linear flux defined (no reaction and source term). Finally the ODE left is defined with a reaction term.


During this project we had time to investigate the implementation of the 1D problem and to start looking at the 2D problem.
We impleted the problem by using the VoronoiFVM.jl package, which is a solver for coupled nonlinear partial differential equations based on the Voronoi finite volume method.

To facilitate the implementation with the VoronoiFVM package, we can rewrite the equations to have the flux and reaction terms on the left hand side and the source on the right hand side. We obtain :


$\frac{\partial u}{\partial t} - \nabla\cdot (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e}) - \frac{1}{\epsilon} (u - \frac{u^{3}}{3} - v) = 0   \;$

$-(-\nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e})) = 0 \;$

$\frac{\partial v}{\partial t} - \epsilon (u + \beta - \gamma v) = 0\;$

since the operator $\nabla \cdot \;$  is linear.

"""

# ╔═╡ c7ff0d86-8e24-11eb-109d-339fdf91ba20


# ╔═╡ 4a2d8d02-8c98-11eb-2b3a-35c636ff9050
md"""
## 2. Implementation of the numerical methods
\

To solve the problem described in the previous section, we can't find an analytical solution so we have to set up a numerical method which will compute an approximate function using a numerical method. We chose to use the voronoi finite volume method to solve our problem.

First we have to define a discretization. The idea is that instead of solving a continuous problem, we solve a large algebraic system called the discrete problem, i.e. we search the value of the unknown functions in a large number of points.

To obtain this discrete problem we're using the FVM which has the advantage of conserving local flux.

Moreover since one of the partial differential equations of our system depends on time, we will have to discretize in time and in space. To do that, we will use the Rothe method, which consists to first discretize in time and then in space.
"""

# ╔═╡ 8843f6f0-8c9b-11eb-39d1-616869eed7c3


# ╔═╡ 36d7a03a-8cac-11eb-3f1a-11c2572fd236
md"""
### 2.1 Finite volume space discretization approach

\
We approximate the solution $u, u_e \text{ and } v$ (our three species) from the bidomain model only with respect to the space discretization. This will result in a spatial semi-discretization.
We optain this spatial semi-discretization of the bidomain model through the voronoi finite volume method.

First we need to generate a mesh. This will be done by the package [ExtendableGrids.jl](https://github.com/j-fu/ExtendableGrids.jl). 

For the 1D case, we create a 1D grids from a vector of monotonicaly increasing x-axis positions. We obtain :

"""

# ╔═╡ aca883f2-8ee2-11eb-0ad2-fd90f6b3a0bf
md"""
 Then for the 2D case, we create a 2D tensor product grids from two vectors of x and y coordinates. This results in the creation of a grid of quadrilaterals. Then, each of them is subdivided into two triangles, resulting in a boundary conforming Delaunay grid. We obtain : 
"""

# ╔═╡ 4acb6716-8c98-11eb-224f-4176a16cf4f2
# Create discretization grid in 1D or 2D with approximately n nodes
function create_grid(n,dim, spatial_domain)
	nx=n
	y_coords = 5
	if dim==2
		nx=ceil(sqrt(n))
		#nx=ceil(n / y_coords)
	end
	X=collect(0:spatial_domain/nx:spatial_domain)
	if dim==1
      grid=simplexgrid(X)
	else
	  #Y=collect(0:1.0:y_coords)
      #grid=simplexgrid(X,Y)

      grid=simplexgrid(X,X)
	end
	return grid,X
end

# ╔═╡ 0dc3e42c-8ee1-11eb-17b9-03754ec9ccda
gridplot(create_grid(10, 1, 10)[1],Plotter=PyPlot,resolution=(600,200),title="uoa")

# ╔═╡ cd9c11e2-8ee1-11eb-1e60-45c96d395944
gridplot(create_grid(10, 2, 10)[1],Plotter=PyPlot,resolution=(600,200),legend_location=(1.5,0))

# ╔═╡ 378e3086-8ee2-11eb-0119-07902cfcac09
md"""

Once we subdivided the computational domain into a finite number of edge (in 1D) or triangle (in 2D), we can build the Voronoi cells which define the representative elementary volumes (or control volumes) for our finite volume method. These voronoi cells respect particular properties which will be useful to solve our problem (e.g. orthogonality of the interfaces between two collation points).

The collocation points of our control volumes are defined as the vertices from the mesh we generated earlier.

Our computational domain is now subdivided in a finite number of control volumes.

We can start by discretizing the continuity equation in each PDEs of the system. We rewrite our system as following: 

$u_t - \nabla\cdot \overrightarrow{j_1} + r_1(u,v) = 0   \;\quad\textbf{(1)}$ 
$\begin{align}
\text{ with } \overrightarrow{j_1} =\sigma_i(\nabla u + \nabla u_e) \; \; \text{and} \;\;\displaystyle r_1(u,v) = -\frac{1}{\epsilon} f(u,v)\end{align}$

$-\nabla\cdot \overrightarrow{j_2} = 0 \;\quad\textbf{(2)}$
$\begin{align}
\text{ with } \overrightarrow{j_2} =\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e} \;\end{align}$

$\displaystyle u_t + r_2(u,v) = 0\;\quad\textbf{(3)}$
$\begin{align}
\text{ with } \displaystyle r_2(u,v) = -\epsilon g(u,v)\end{align}$


Then we integrate the spatial derivative over the control volumes $\omega_k$.\
For the equation (1), we have : 

$\displaystyle u_t - \int_{\omega_k} \nabla \overrightarrow{j_1} \; \mathrm{d}\omega + \int_{\omega_k} r_1(u,v) \; \mathrm{d}\omega = 0 \;$

$\Rightarrow \displaystyle u_t - \int_{\partial\omega_k} \overrightarrow{j_1} \cdot \overrightarrow{n}_{\omega} \; \mathrm{d}s + \int_{\omega_k} r_1(u,v) \; \mathrm{d}\omega = 0 \;\quad\text{ (by Gauss' law)}$

$\Rightarrow \displaystyle u_t - \sum_{l\in N_k}\int_{\sigma_{kl}} \overrightarrow{j_1} \cdot \overrightarrow{\nu}_{kl} \; \mathrm{d}s + \sum_{m\in G_k}\int_{\gamma_{kl}} \overrightarrow{j_1} \cdot \overrightarrow{n}_{m} \; \mathrm{d}s + \int_{\omega_k} r_1(u,v) \; \mathrm{d}\omega = 0 \;\quad$

For the equation (2), we have : 

$-\displaystyle \int_{\omega_k} \nabla \overrightarrow{j_2} \; \mathrm{d}\omega = 0 \;$

$\Rightarrow \displaystyle - \int_{\partial\omega_k} \overrightarrow{j_2} \cdot \overrightarrow{n}_{\omega} \; \mathrm{d}s = 0 \;\quad\text{ (by Gauss' law)}$

$\Rightarrow \displaystyle - \sum_{l\in N_k}\int_{\sigma_{kl}} \overrightarrow{j_2} \cdot \overrightarrow{\nu}_{kl} \; \mathrm{d}s - \sum_{m\in G_k}\int_{\gamma_{kl}} \overrightarrow{j_2} \cdot \overrightarrow{n}_{m} \; \mathrm{d}s = 0 \;\quad$

For the equation (3), we have : 

$\displaystyle u_t + \int_{\omega_k} r_2(u,v) \; \mathrm{d}\omega = 0 \;$

We are able, thanks to the collocation points, to approximate the fluxes by finite difference using the neigboring control volumes.
We define flux functions : 

$\text{for }\overrightarrow{j_1} =\sigma_i(\nabla u + \nabla u_e), \text{we use } g_1(u^k,u^l,u_e^k,u_e^l) = \sigma_i(u^k-u^l)+\sigma_i(u_e^k-u_e^l)$

$\text{for }\overrightarrow{j_2} =\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}, \text{we use } g_2(u^k,u^l,u_e^k,u_e^l) = \sigma_i(u^k-u^l)+(\sigma_i+\sigma_e)(u_e^k-u_e^l)$

These flux functions will be important in the implementation of our problem with the VoronoiFVM.jl package.

"""

# ╔═╡ b7ed7f66-8fca-11eb-3fd3-f3efb6c7ffdb
md"""
Given a control volume $\omega_k$, we now integrate the different equations of the system of PDEs over space-time control volume $\omega_k \times (t^{n-1},t^n),$ divide by $\tau^n :$ \
For the equation (1) :

$\int_{\omega_k}(\frac{\partial u}{\partial t} - \nabla\cdot (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e}) - \frac{1}{\epsilon} f(u,v)) \; \mathrm{d}\omega= 0   \;$

$\Rightarrow \int_{\omega_k}(\frac{1}{\tau^n}(u^n - u^{n-1}) - \nabla\cdot (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e}) - \frac{1}{\epsilon} f(u,v)) \; \mathrm{d}\omega= 0$

$\Rightarrow \frac{1}{\tau^n} \int_{\omega_k}(u^n - u^{n-1})\; \mathrm{d}\omega - \int_{\omega_k} \nabla\cdot (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e})\; \mathrm{d}\omega - \int_{\omega_k} \frac{1}{\epsilon} f(u,v)) \; \mathrm{d}\omega= 0$

$\Rightarrow -\sum_{l \in N_k} \int_{\sigma_{kl}}(\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e})\cdot \vec n_{kl}\; \mathrm{d}\gamma - \int_{\gamma_k} (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e})\; \mathrm{d}\gamma - \int_{\omega_k} \frac{1}{\epsilon} f(u,v)) \; \mathrm{d}\omega$

$+\frac{1}{\tau^n} \int_{\omega_k}(u^n - u^{n-1})\; \mathrm{d}\omega= 0$

$\Rightarrow \underbrace{\frac{\lvert\omega_k\rvert}{\tau^n}(u^n - u^{n-1})}_{\rightarrow M} - \underbrace{\sum_{l \in N_k} \frac{\lvert\sigma_{kl}\rvert}{h_{kl}}
(\sigma_{i}(u^k-u^l + u_e^k -u_e^l)) - \sum_{m\in G_k}\lvert \gamma_{km}\rvert(\alpha_m u^k +\beta_m u_e^k)}_{\rightarrow A}$

$\underbrace{- \frac{\lvert\omega_k\rvert}{\epsilon} f(u,v)}_{F(u,v)}$

$\Rightarrow \frac{1}{\tau^n}(Mu^n - Mu^{n-1}) - A(u + u_e) + F(u,v) = 0$

For the equation (2):

$\int_{\omega_k} \nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}) \; \mathrm{d}\omega= 0$

$\Rightarrow \underbrace{\sum_{l\in N_k} \frac{\lvert\sigma_{kl}\rvert}{h_{kl}}\sigma_i(u^k - u ^l) + \sum_{m\in G_k} \lvert\gamma_{km}\rvert(\alpha_m u^k)}_{\rightarrow A_1} + \underbrace{\sum_{l\in N_k} \frac{\lvert\sigma_{kl}\rvert}{h_{kl}}(\sigma_i + \sigma_e)(u^k_e - u ^l_e) + \sum_{m\in G_k} \lvert\gamma_{km}\rvert(\alpha_m u_e^k)}_{\rightarrow A_2}$

$\Rightarrow A_1 u + (A_1 + A_2)u_e = 0$

For the equation (3) :

$\underbrace{\frac{\lvert\omega_k\rvert}{\tau^n}(v^n - v^{n-1})}_{\rightarrow M_1} + - \epsilon\underbrace{\lvert\omega_k\rvert g(u,v)}_{G(u,v)}$

$\Rightarrow M_1v - \epsilon G(u,v) = 0$

"""

# ╔═╡ d51cc3ac-8ef6-11eb-1c8d-07d812716f2f


# ╔═╡ 7905e330-8d89-11eb-0503-c75a1fbddfe5
md"""
### 2.2 Time discretization approach

\

We now look at the time discretization which will consist of replacing the time derivative in the equations $(1)$ and $(3)$ by a difference quotient.

The strategy to solve a parabolic initial boundary value problem is that we are not trying to solve the whole problem at once but we consider a sequence of solutions by time integrations.

Let $\tau>0$, $\;$  $\displaystyle \tau=\frac{T}{N_k}, \; N_k \in \mathbb{N} \; \;$ be the time stepsize. 

For i = 1 ... $N_k \;\;$, solve

$\displaystyle \frac{u^n - u^{n-1}}{\tau} - \nabla\cdot (\sigma_{i} \nabla u^{\theta} + \sigma_{i} \nabla u_{e}^{\theta}) - \frac{1}{\epsilon} (u^{\theta} - \frac{(u^{3})^{\theta}}{3} - v^{\theta}) = 0   \;$

$-\nabla\cdot (\sigma_{i} \nabla u^{\theta} + (\sigma_{i} + \sigma_{e}) \nabla u_{e}^{\theta}) = 0 \;$

$\displaystyle \frac{u^n - u^{n-1}}{\tau} - \epsilon (u^{\theta} + \beta - \gamma v^{\theta}) = 0\;$

with $u^{\theta} = \theta u^n + (1- \theta)u^{n-1} \; , \; u_e^{\theta} = \theta u_e^n + (1- \theta)u_e^{n-1} \; \text{and} \; v^{\theta} = \theta v^n + (1- \theta)v^{n-1}$

We can already notice that the equation (2) of the system doesn't depend on time, so we will have to solve a system equations for each step time. \
There is several possibilities to conduct the time discretization on our equation (1) and (3).
We chose to look at the $\theta$-scheme. The $\theta$-scheme has three special cases : 
- for $\theta = 1$ : backward (implicit) Euler method
- for $\theta = \frac{1}{2}$ : Crank-Nicolson scheme
- for $\theta = 0$ : forward (explicit) Euler method

We present the backward Euler method :

$M\frac{u^{n+1}-u^n}{\Delta t} - A(u^{n+1} + u_e^{n+1})+F(u^{n+1},v^{n+1}) = 0$

$M\frac{v^{n+1}-v^n}{\Delta t}- \epsilon G(u^{n+1},v^{n+1})$

with $\Delta t$ defined as a constant time step. 

"""

# ╔═╡ bdfe4cac-8e15-11eb-04b2-172cb6acaec1
md"""
## 3. Numerical solutions methods

We performed our space and time discretization of our computational domain.
We can now solve the system our bidomain equations.
We look at the full discretization which consists of combining the voronoi finite volume method and some Euler-scheme (here we chose the backward euler method).


The backward and forward Euler method have a first order convergence in time and the CN method has a second order convergence in time. Moreover for the backward and CN method, to compute the solution for one time step we have to solve one system of equations, this is not the case with the forward Euler method (only a product matrix vector).

The explicit Euler method could be consider since it’s the same order of convergence as the implicit Euler method but there is an advantage to use this scheme because we don’t have to solve a system of equations to get a solution.

If we investigate the $\theta$-scheme with respect to the maximum norm, we can study the stability and see that the method is stable in the case :

$(1-\theta) \tau \leq \frac{1}{2} h^2$

This stability condition is called the Courant Friedrichs Lewy (CFL)  condition for parabolic PDEs.

The cheap explicit Euler method only works for very small time steps. That is a compensation with respect to the fact that we don’t have to solve a linear system.


We chose to use a backward Euler scheme so we don’t worry about the stability of the scheme since this implicit scheme is unconditionally stable.


"""

# ╔═╡ 06c0fe66-8e17-11eb-1b01-c5f153f6fc4c
md"""
## 4. Results of the Bidomain Model
"""

# ╔═╡ 3708363c-8cac-11eb-29ed-4b03f07cb1cd
md"""
We present in this section the numerical results we obtained for the 1D case. We will look at the 1D results on a 1D grid and then the 1D results on a 2D grid. \
We used the [VoronoinFVM.jl](https://github.com/j-fu/VoronoiFVM.jl) package to solve the bidomain problem for the 1D case.
"""

# ╔═╡ 805e0534-8e17-11eb-10c7-bdd096781dca
md"""
### 4.1 Solve 1D
"""

# ╔═╡ 2a4810e8-8f01-11eb-0bbc-595db6c6c842
md"""
We solved the 1D case of the bidomain model on the domain $[0,L]$ , with $L = 70$ . \
The 1D bidomain model is written as :

$\frac{\partial u}{\partial t} - \frac{\partial}{\partial x} (\sigma_{i} \frac{\partial u}{\partial x} + \sigma_{i} \frac{\partial u_e}{\partial x}) - \frac{1}{\epsilon} (u - \frac{u^{3}}{3} - v) = 0   \;$

$-(-\frac{\partial}{\partial x} (\sigma_{i} \frac{\partial u}{\partial x} + (\sigma_{i} + \sigma_{e}) \frac{\partial u_e}{\partial x})) = 0 \;$

$\frac{\partial v}{\partial t} - \epsilon (u + \beta - \gamma v) = 0\;$

We implemented this system with homogeneous Neumann boundary conditions such that $\displaystyle \partial_x u(0) = \partial_x u(L) = 0$  and  $\displaystyle \partial_x u_e(0) = \partial_x u_e(L) = 0$. 

Since we have pure Neumann boundary conditions, we add a supplementary condition otherwise the linear system of equations to solve would be singular. We define the homogeneous Dirichlet boundary condition $u_e(0)=0$.

We introduced the constant from $[1]$ , i.e. $\epsilon = 0.1, \beta = 1.0 ,$ $\gamma = 0.5$ and the conductivity tensors (in 1D, scarlars) $\sigma_i = \sigma_e = 1.0$. 

We implemented the discrete problem from the section 2 with the VoronoiFVM.jl package. It is described in the functions storage, reaction and flux. Since we have a discrete problem with some nonlinearity, the VoronoiFVM.jl package is using the Newton iteration scheme to solve the nonlinear system of equations.

In a first part, we solved the stationary problem, i.e. without the time derivative component.

"""

# ╔═╡ 856e31f8-8f9e-11eb-10a9-5ba1e75784da
md"""
Later we looked at the unstationary problem. We define $T = 50$ which is the time end of the evolution of our system of PDEs. We chose this time so we have time to see the system reach the stationary state. \
We take the species $u$ and $v$ at the equilibrium value of the system for the initial conditions of our time dependent problems.

$\begin{cases}
      f(u,v) = 0 \\
      g(u,v) = 0
\end{cases}$

$\displaystyle \Rightarrow \begin{cases}
      u-\frac{u^3}{3} -v = 0 \\
      u + \beta - \gamma v = 0
\end{cases} \quad \;$

So we will have to solve a nonlinear system of equations to find the initial value of the specie $u$ and $v$.
For the species $u_e$, we choose u_e constant at 0, except for the intervall $[0, \frac{1}{20}L]$ where we apply the excitation so that $u(x) = 2$.
"""

# ╔═╡ 65688fdc-8f9e-11eb-0b23-9175bd3666dc
md"""
We computed the solution for a rather coarse grid (N = 0.07) with the backward Euler method. We chose a arbitrary time step dt = 0.01.

After several tests, we can see that we obtain a numerical solution for different
grid sizes and time steps. The parameter dtgrowth from our implementation, which is defining how fast we increase the timestep (by factor) when approaching stationary is important because if it's above 1.005, we notice that we obtain some convergence error.
"""

# ╔═╡ 87f04ab4-8e17-11eb-2a54-71d72b8f8469
md"""
### 4.2 Solve 1D on a 2D grid
"""

# ╔═╡ 2b0d41d6-8f01-11eb-182a-5df8ced7038f
md"""
We take the same constants as for the previous test. We only change the grid where we solve the problem. This time we take a 2D tensor product mesh.

For the stationary problem only the grid changes.
"""

# ╔═╡ d59380ea-8f9f-11eb-2235-f749743155c7
md"""
For the unstationary problem we need to modify the initial value, since now we have to take the interval $[0, \frac{1}{20}L]$ in the both directions (x-axis and y-axis).

As result for the backward Euler scheme, we obtain :
"""

# ╔═╡ 333a5e58-8fa0-11eb-174a-5961cffddf4a


# ╔═╡ 902feeb4-8e17-11eb-3c76-f1b04999e0b5
md"""
## 5. Performance improvement
"""

# ╔═╡ c2a06b38-8f13-11eb-0c5b-c169c89d9778
md"""
Improve the performance to solve one model is important, so we can have better accuracy and a faster time to solve the equations.

In our case, several tracks could be study to improve the performances.
We could look into different time discretization to see if the system is solved faster. We could also look into second or third order scheme for a better accuracy of the solutions.

From [1], we can see that a semi-implicit method would be a good idea to implement. It would enable us to obtain a good numerical solution without worrying about stability conditions but also these methods don't need to solve a system of equations for each time step.
"""

# ╔═╡ 7996bbc4-8fdc-11eb-3e65-63e913138e2b


# ╔═╡ dbf6a9a2-8f9e-11eb-1811-41bc06287e1c
md"""
## 6. Julia functions
"""

# ╔═╡ 79d48610-8f9c-11eb-0639-a3e94be8e801
# Function describing evolution of system with initial value inival 
# using the Implicit Euler method
function evolution(inival, # initial value
		           sys,    # finite volume system
		           grid,   # simplex grid
		           tstep,  # initial time step 
		           tend,   # end time 
		           dtgrowth  # time step growth factor
	               )
	time=0.0
	# record time and solution
	times=[time]
	solutions=[copy(inival)]
	
	solution=copy(inival)
    while time<tend
        time=time+tstep
        solve!(solution,inival,sys,tstep=tstep) # solve implicit Euler time step 
        inival.=solution  # copy solution to inivalue
       	push!(times,time) 
		push!(solutions,copy(solution))
        tstep*=dtgrowth  # increase timestep by factor when approaching stationary state
    end
	# return result and grid 
	(times=times,solutions=solutions,grid=grid)
end

# ╔═╡ 96a39c9c-8f9c-11eb-30ec-2fd05485e83b
grid1d_a = create_grid(1000,1, 70)[1]

# ╔═╡ 8eed0592-8f9c-11eb-33eb-af70ce2e76cf
function bidomain_stationary(grid; sigma_i=1.0, sigma_e=1.0, epsilon=0.1, gamma=0.5, beta=1)

	function bidomain_flux!(f,_u,edge)
		u=unknowns(edge,_u)
		
		f[1] = (sigma_i * (u[1,1] - u[1, 2]) + sigma_i * (u[2,1] - u[2,2]))
		
		f[2] = -sigma_i * (u[1,1] - u[1, 2]) - (sigma_i + sigma_e) * (u[2,1]-u[2,2])
	end

	function bidomain_reaction!(f,u,node)
		f[1] = -(1 / epsilon) *  (u[1]  - (u[1] ^ 3) / 3 - u[3])

		f[3] = -epsilon * (u[1]  + beta - gamma * u[3])
	end

	# Create system
	bidomain_physics=VoronoiFVM.Physics(flux=bidomain_flux!,
									 num_species=3,reaction=bidomain_reaction!)
	
	bidomain_system=VoronoiFVM.DenseSystem(grid,bidomain_physics)

	enable_species!(bidomain_system,1,[1])
	enable_species!(bidomain_system,2,[1])
	enable_species!(bidomain_system,3,[1])
	
	west = dim_space(grid)==1  ? 1 : 4

	# Dirichlet to set u_e = 0 at index 0
	boundary_dirichlet!(bidomain_system, 2, west, 0)

	solve(unknowns(bidomain_system,inival=0),bidomain_system)
end

# ╔═╡ a3031148-8f9c-11eb-328b-11855a556a81
result_bidomain_stationary = bidomain_stationary(grid1d_a)

# ╔═╡ bc1f2e80-8f9e-11eb-298b-4f349573cbab
let
	bivis=GridVisualizer(layout=(1,3),resolution=(600,300),Plotter=PyPlot)
	scalarplot!(bivis[1,1],grid1d_a,
	       result_bidomain_stationary[1,:],
		   title="u",
	       flimits=(-2,2),colormap=:cool,levels=50,clear=true)
	scalarplot!(bivis[1,2],grid1d_a,
	       result_bidomain_stationary[2,:],
		   title="u_e",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
	scalarplot!(bivis[1,3],grid1d_a,
	       result_bidomain_stationary[3,:],
		   title="v",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
end

# ╔═╡ c69754a2-8f9c-11eb-0e31-f96b4b9d3f13
grid2d_a = create_grid(1000,2, 70)[1]

# ╔═╡ 0ece6fa6-8f9f-11eb-2d3a-953f67663fca
result_bidomain_stationary1d_2dgrid = bidomain_stationary(grid2d_a)

# ╔═╡ 06e3a6ee-8f9f-11eb-2420-8fe02d48fa7f
let
	bivis=GridVisualizer(layout=(1,3),resolution=(600,300),Plotter=PyPlot)
	scalarplot!(bivis[1,1],grid2d_a,
	       result_bidomain_stationary1d_2dgrid[1,:],
		   title="u",
	       flimits=(-2,2),colormap=:cool,levels=50,clear=true)
	scalarplot!(bivis[1,2],grid2d_a,
	       result_bidomain_stationary1d_2dgrid[2,:],
		   title="u_e",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
	scalarplot!(bivis[1,3],grid2d_a,
	       result_bidomain_stationary1d_2dgrid[3,:],
		   title="v",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
end

# ╔═╡ dee1486a-8f9c-11eb-1951-d54a4cd4b0ac
function bidomain(;n=100, sd=70,dim=1,sigma_i=1.0, sigma_e=1.0, epsilon=0.1, gamma=0.5, beta=1, tstep=0.01, tend=50,dtgrowth=1.005)
	
	grid, L =create_grid(n,dim, sd)
	
	function storage!(f,u,node)
		# Set all indices of f to values in u
        f[1] = u[1]
		f[2] = 0
		f[3] = u[3]
    end

	
	function bidomain_flux!(f,_u,edge)
		u=unknowns(edge,_u)
		# u
		f[1] = sigma_i * (u[1,1] - u[1, 2]) + sigma_i * (u[2,1] - u[2,2])
		# u_e
		f[2] = -sigma_i * (u[1,1] - u[1, 2]) - (sigma_i + sigma_e) * (u[2,1]-u[2,2])
		# v

	end
	# Reaction:
	function bidomain_reaction!(f,u,node)
		f[1] = (-1 / epsilon) *  (u[1]  - (u[1] ^ 3) / 3 - u[3])
		f[3] = - 1 * epsilon * (u[1]  + beta - gamma * u[3])
	end


	# Create system
	bidomain_physics=VoronoiFVM.Physics(flux=bidomain_flux!,storage=storage!,
									 num_species=3,reaction=bidomain_reaction!, 	
		)
	bidomain_system=VoronoiFVM.DenseSystem(grid,bidomain_physics)

	enable_species!(bidomain_system,1,[1])
	enable_species!(bidomain_system,2,[1])
	enable_species!(bidomain_system,3,[1])

	west = dim_space(grid)==1  ? 1 : 4

	# Dirichlet to set u_e = 0 at index 0
	boundary_dirichlet!(bidomain_system, 2, west, 0)
	

	inival=unknowns(bidomain_system)
	
	# We solve the equilibriam of the system, aka where f and g are 0
	 function f!(F, v)
		u = v[1]
		v = v[2]
		F[1] = u - (u^3)/3 - v
		F[2] = u + beta - gamma * v
	end
 
 	res = nlsolve(f!, [0.0; 0.0])
	u_init = res.zero[1]
	v_init = res.zero[2]


	for i=1:num_nodes(grid)
		x_coord = (i - 1) % length(L) + 1
		y_coord = convert(Int64, ceil(i / length(L)))
		if L[x_coord] < sd / 20 && (dim == 1 || L[y_coord] < sd / 20)
		# We set the initial value to 2 if within the first 1/20th of the grid, as specified by the paper
			inival[1,i]= 2
		else
			inival[1,i]= u_init
		end



		inival[2,i]= 0
		inival[3,i]= v_init
	end



	evolution(inival,bidomain_system,grid,tstep,tend,dtgrowth)	
end



# ╔═╡ ed62389a-8f9c-11eb-0672-9723bbfdf2a8
result_bidomain=bidomain(n=1000,sd=70.0,dim=1)

# ╔═╡ 7b958a1c-8f9e-11eb-1095-7f99f3409191
md"""
time=$(@bind t_bidomain Slider(1:length(result_bidomain.times),default=450))
"""

# ╔═╡ 6412bdce-8f9e-11eb-21f4-0986129c19fa
let
	bivis=GridVisualizer(layout=(1,3),resolution=(600,300),Plotter=PyPlot)
	scalarplot!(bivis[1,1],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][1,:],
		   title="u: t=$(round(result_bidomain.times[t_bidomain], digits=6))",
	       flimits=(-2,2),colormap=:cool,levels=50,clear=true)
	scalarplot!(bivis[1,2],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][2,:],
		   title="u_e: t=$(round(result_bidomain.times[t_bidomain], digits=6))",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
	scalarplot!(bivis[1,3],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][3,:],
		   title="v: t=$(round(result_bidomain.times[t_bidomain], digits=6))",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
end

# ╔═╡ 3622c982-8f9d-11eb-3632-e7b5302793ef
result_bidomain_1d_2dgrid=bidomain(n=1000,sd=20,dim=2)

# ╔═╡ 4c61a920-8f9d-11eb-3c46-31887c3cca4f
md"""
time=$(@bind t_bidomain_1d_2dgrid Slider(1:length(result_bidomain_1d_2dgrid.times),default=400))
"""

# ╔═╡ 5bd2bf0c-8f9d-11eb-35de-cf995ed7f32e
let
		bivis=GridVisualizer(layout=(1,3),resolution=(600,300),Plotter=PyPlot)
		scalarplot!(bivis[1,1],result_bidomain_1d_2dgrid.grid,
			   result_bidomain_1d_2dgrid.solutions[t_bidomain_1d_2dgrid][1,:],
			   title="u: t=$(round(result_bidomain_1d_2dgrid.times[t_bidomain_1d_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,clear=true)
		scalarplot!(bivis[1,2],result_bidomain_1d_2dgrid.grid,
			   result_bidomain_1d_2dgrid.solutions[t_bidomain_1d_2dgrid][2,:],
			   title="u_e: t=$(round(result_bidomain_1d_2dgrid.times[t_bidomain_1d_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,show=true)
		scalarplot!(bivis[1,3],result_bidomain_1d_2dgrid.grid,
			   result_bidomain_1d_2dgrid.solutions[t_bidomain_1d_2dgrid][3,:],
			   title="v: t=$(round(result_bidomain_1d_2dgrid.times[t_bidomain_1d_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,show=true)
	end

# ╔═╡ 79b3f5e2-8d89-11eb-0f7f-a94b158e574a
md"""
## References
[1] _Semi-implicit time-discretization schemes for the bidomain model_ by Marc Ethier and Yves Bourgault.

[2] _Existence and uniqueness of the solution for the bidomain model used in cardiac electrophysiology_ by Yves Bourgault, Yves Coudière, Charles Pierre.
"""

# ╔═╡ Cell order:
# ╟─0a475948-8c98-11eb-0787-2546680a44e6
# ╟─d846fa24-8c98-11eb-3524-8d570cf24952
# ╟─496173c0-8c98-11eb-086e-b56945782ecf
# ╟─c7ff0d86-8e24-11eb-109d-339fdf91ba20
# ╟─4a2d8d02-8c98-11eb-2b3a-35c636ff9050
# ╟─8843f6f0-8c9b-11eb-39d1-616869eed7c3
# ╟─36d7a03a-8cac-11eb-3f1a-11c2572fd236
# ╟─0dc3e42c-8ee1-11eb-17b9-03754ec9ccda
# ╟─aca883f2-8ee2-11eb-0ad2-fd90f6b3a0bf
# ╟─cd9c11e2-8ee1-11eb-1e60-45c96d395944
# ╟─4acb6716-8c98-11eb-224f-4176a16cf4f2
# ╟─378e3086-8ee2-11eb-0119-07902cfcac09
# ╟─b7ed7f66-8fca-11eb-3fd3-f3efb6c7ffdb
# ╟─d51cc3ac-8ef6-11eb-1c8d-07d812716f2f
# ╟─7905e330-8d89-11eb-0503-c75a1fbddfe5
# ╟─bdfe4cac-8e15-11eb-04b2-172cb6acaec1
# ╟─06c0fe66-8e17-11eb-1b01-c5f153f6fc4c
# ╟─3708363c-8cac-11eb-29ed-4b03f07cb1cd
# ╟─805e0534-8e17-11eb-10c7-bdd096781dca
# ╟─2a4810e8-8f01-11eb-0bbc-595db6c6c842
# ╟─bc1f2e80-8f9e-11eb-298b-4f349573cbab
# ╟─856e31f8-8f9e-11eb-10a9-5ba1e75784da
# ╟─7b958a1c-8f9e-11eb-1095-7f99f3409191
# ╟─6412bdce-8f9e-11eb-21f4-0986129c19fa
# ╟─65688fdc-8f9e-11eb-0b23-9175bd3666dc
# ╟─87f04ab4-8e17-11eb-2a54-71d72b8f8469
# ╟─2b0d41d6-8f01-11eb-182a-5df8ced7038f
# ╟─06e3a6ee-8f9f-11eb-2420-8fe02d48fa7f
# ╟─d59380ea-8f9f-11eb-2235-f749743155c7
# ╟─333a5e58-8fa0-11eb-174a-5961cffddf4a
# ╟─4c61a920-8f9d-11eb-3c46-31887c3cca4f
# ╟─5bd2bf0c-8f9d-11eb-35de-cf995ed7f32e
# ╟─902feeb4-8e17-11eb-3c76-f1b04999e0b5
# ╟─c2a06b38-8f13-11eb-0c5b-c169c89d9778
# ╟─7996bbc4-8fdc-11eb-3e65-63e913138e2b
# ╟─dbf6a9a2-8f9e-11eb-1811-41bc06287e1c
# ╟─79d48610-8f9c-11eb-0639-a3e94be8e801
# ╟─96a39c9c-8f9c-11eb-30ec-2fd05485e83b
# ╟─8eed0592-8f9c-11eb-33eb-af70ce2e76cf
# ╟─a3031148-8f9c-11eb-328b-11855a556a81
# ╟─c69754a2-8f9c-11eb-0e31-f96b4b9d3f13
# ╟─0ece6fa6-8f9f-11eb-2d3a-953f67663fca
# ╟─dee1486a-8f9c-11eb-1951-d54a4cd4b0ac
# ╟─ed62389a-8f9c-11eb-0672-9723bbfdf2a8
# ╟─3622c982-8f9d-11eb-3632-e7b5302793ef
# ╟─79b3f5e2-8d89-11eb-0f7f-a94b158e574a
# ╟─f0083160-8c97-11eb-04c0-ad918b035d75
