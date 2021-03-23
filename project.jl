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

# ╔═╡ 60941eaa-1aea-11eb-1277-97b991548781
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

# ╔═╡ 48b1a0ac-76f3-11eb-05bd-cbcfae8e2f27
md"""
# Bidomain Model
  Gregoire Pourtier & Conrad Schloer
"""

# ╔═╡ 397c9290-76f5-11eb-1114-4bd31f7ecf9a
	md"""
## 1. Problem Overview


Equations for the Bidomain problem, a three species problem


$\frac{\partial u}{\partial t} = \frac{1}{\epsilon} f(u ,v) +  \nabla\cdot (\sigma_{i} \nabla u) + \nabla\cdot (\sigma_{i} \nabla u_{e})\;$

$\nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}) = 0 \;$

$\frac{\partial v}{\partial t} = \epsilon g(u, v)\;$

where

$f(u, v)= u−\frac{u^{3}}{3}−v\;$

$g(u, v)= u + \beta - \gamma v \;$


We can therefore rewrite the equations as


$\frac{\partial u}{\partial t} - \nabla\cdot (\sigma_{i} \nabla u + \sigma_{i} \nabla u_{e}) - \frac{1}{\epsilon} (u - \frac{u^{3}}{3} - v) = 0   \;$

$\nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}) = 0 \;$

$\frac{\partial v}{\partial t} - \epsilon (u + \beta - \gamma v) = 0\;$


"""

# ╔═╡ 90328ff6-8643-11eb-0f55-314c878ba3ec
md"""
## 2. Implementation

We define evolution and create_grid similar to the lecture, though the grid now goes to 70 instead of 1.

"""


# ╔═╡ 95a667de-880d-11eb-0171-b93ed1f38ea1
spatial_domain = 20.0

# ╔═╡ 633b3d12-76a4-11eb-0bc7-b9bf9116933f
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


# ╔═╡ 4b9f5030-76cc-11eb-117c-91ca8336c30b
# Create discretization grid in 1D or 2D with approximately n nodes
function create_grid(n,dim)
	nx=n
	if dim==2
		nx=ceil(sqrt(n))
	end
	X=collect(0:spatial_domain/nx:spatial_domain)
	if dim==1
      grid=simplexgrid(X)
	else
      grid=simplexgrid(X,X)
	end
	return grid,X
end

# ╔═╡ 023173fe-8644-11eb-3303-e351dbf44aaf
# We show an example grid, for the 1 dimensional problem
gridplot(create_grid(10, 1)[1],Plotter=PyPlot,resolution=(600,200))

# ╔═╡ 7278ba0a-8b00-11eb-3629-e55ab965940c


# ╔═╡ 3402cd3c-8afc-11eb-2af1-312ae538cd1a
md"""
### 2.1 Solve the stationary problem
"""

# ╔═╡ 82ed33a0-8b00-11eb-11d0-cddce2e38e2c
md"""
Grid in domain $\Omega=(0,70)$ consisting of N=$(@bind N Scrubbable(500:100:2000,default=1000)) points.
"""

# ╔═╡ 990dd67c-8afc-11eb-0f5d-f1525f921906
grid1d_a = create_grid(N,1)[1]

# ╔═╡ 50bc7ea0-8afc-11eb-1101-d7a7373ed0ce
function bidomain_stationary(grid; sigma_i=1.0, sigma_e=1.0, epsilon=0.1, gamma=0.5, beta=1)

	function bidomain_flux!(f,_u,edge)
		u=unknowns(edge,_u)
		
		f[1] = (sigma_i * (u[1,1] - u[1, 2]) + sigma_i * (u[2,1] - u[2,2]))
		
		f[2] = sigma_i * (u[1,1] - u[1, 2]) + (sigma_i + sigma_e) * (u[2,1]-u[2,2])
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

	# Dirichlet to set u_e = 0 at index 0
	boundary_dirichlet!(bidomain_system, 2, 1, 0)

	solve(unknowns(bidomain_system,inival=0),bidomain_system)
end

# ╔═╡ 5f7bbdc0-8afc-11eb-1b38-e3448733ad4b
result_bidomain_stationary = bidomain_stationary(grid1d_a);

# ╔═╡ 81f270e2-8afc-11eb-24be-5952f95e6aa3
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

# ╔═╡ cbb19904-8afc-11eb-19a5-47ad0bae2bfd
md"""
### 2.2 Solve the unstationary problem
"""

# ╔═╡ b1a3c0a6-8643-11eb-1a7b-cd4720e77617
md"""
Now, we create the bidomain function with flux and reaction.

"""


# ╔═╡ fa52bcd0-76f8-11eb-0d58-955a514a00b1
function bidomain(;n=100,dim=1,sigma_i=1.0, sigma_e=1.0, epsilon=0.1, gamma=0.5, beta=1, tstep=0.01, tend=50,dtgrowth=1.005)
	
	grid, L =create_grid(n,dim)
	
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
		f[2] = sigma_i * (u[1,1] - u[1, 2]) + (sigma_i + sigma_e) * (u[2,1]-u[2,2])
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
	
	if dim==2
	
		lex_ordering = Float64[]
		for i=1:length(L)
			for j=1:length(L)
				if (L[i]<spatial_domain / 20) && (L[j]<spatial_domain / 20)
					append!(lex_ordering,(L[i] ,L[j]))	
				else
					append!(lex_ordering,(-1,-1))
				end
			end
		end

		lex_ordering1 = reshape(lex_ordering,(2,num_nodes(grid)))'
	
	
		for i=1:num_nodes(grid)
		# We set the initial value to 2 if within the first 1/20th of the grid, as 	specified by the paper

			if max(lex_ordering1[i,1],lex_ordering1[i,2])!=-1 #L[(i - 1) % length(L) + 1]< spatial_domain / 20
				inival[1,i]= 2
			else
				inival[1,i]= u_init
			end


			inival[2,i]= 0
			inival[3,i]= v_init

		end
		
	# In the 1d case
	else
		for i=1:num_nodes(grid)
	# We set the initial value to 2 if within the first 1/20th of the grid, as specified by the paper

			if L[i]< spatial_domain / 20
				inival[1,i]= 2
			else
				inival[1,i]= u_init
			end



			inival[2,i]= 0
			inival[3,i]= v_init
		end
	end
	


	evolution(inival,bidomain_system,grid,tstep,tend,dtgrowth)	
end


# ╔═╡ 4e66a016-76f9-11eb-2023-6dfc3374c066
result_bidomain=bidomain(n=100,dim=1);

# ╔═╡ 106d3bc0-76fa-11eb-1ee6-3fa73be52226
md"""
time=$(@bind t_bidomain Slider(1:length(result_bidomain.times),default=1))
"""

# ╔═╡ e2cbc0ec-76f9-11eb-2870-f10f6cdc8be4
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

# ╔═╡ 9449905c-8b15-11eb-0987-471b19ff966b
md"""
### 2.3 Solving the problem with a 2D grid
"""

# ╔═╡ a186f7f2-8b15-11eb-195d-5fe71ec9fd1e
gridplot(create_grid(10, 2)[1],resolution=(600,200),Plotter=PyPlot,legend_location=(1.5,0))

# ╔═╡ 46e7c83a-8bdd-11eb-211d-31a8a2b766c3
md"""
Plot Bidomain 2D ?
$@bind do_2d_plot CheckBox(default=false))
"""

# ╔═╡ 435e9954-8b16-11eb-06fa-f70df37efee9
if do_2d_plot
	result_bidomain_2dgrid=bidomain(n=1000,dim=2);
end;

# ╔═╡ 5125d26e-8b16-11eb-2da0-235368e7840c
if do_2d_plot
	md"""
	time=$(@bind t_bidomain_2dgrid Slider(1:length(result_bidomain_2dgrid.times),default=1))
	"""
end

# ╔═╡ 6e46e702-8b16-11eb-2edf-e12f7c97594d
if do_2d_plot
	let
		bivis=GridVisualizer(layout=(1,3),resolution=(600,300),Plotter=PyPlot)
		scalarplot!(bivis[1,1],result_bidomain_2dgrid.grid,
			   result_bidomain_2dgrid.solutions[t_bidomain_2dgrid][1,:],
			   title="u: t=$(round(result_bidomain_2dgrid.times[t_bidomain_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,clear=true)
		scalarplot!(bivis[1,2],result_bidomain_2dgrid.grid,
			   result_bidomain_2dgrid.solutions[t_bidomain_2dgrid][2,:],
			   title="u_e: t=$(round(result_bidomain_2dgrid.times[t_bidomain_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,show=true)
		scalarplot!(bivis[1,3],result_bidomain_2dgrid.grid,
			   result_bidomain_2dgrid.solutions[t_bidomain][3,:],
			   title="v: t=$(round(result_bidomain_2dgrid.times[t_bidomain_2dgrid], digits=4))",
			   flimits=(-2,2),colormap=:cool,levels=50,show=true)
	end
end

# ╔═╡ 77e8e9ce-8bd0-11eb-0d60-cf66bd337f0c
md"""
Can change flimits $= (-2,2)$ to flimits $= (2,-2)$ for species u and u_e to see differently 
(but time != 0)
"""

# ╔═╡ 3ab28264-6c64-11eb-29f4-a9ed2e9eba16
TableOfContents()

# ╔═╡ d32173ec-66e8-11eb-11ad-f9605b4964b2
with_terminal() do
	Pkg.status()
end

# ╔═╡ Cell order:
# ╠═60941eaa-1aea-11eb-1277-97b991548781
# ╟─48b1a0ac-76f3-11eb-05bd-cbcfae8e2f27
# ╟─397c9290-76f5-11eb-1114-4bd31f7ecf9a
# ╟─90328ff6-8643-11eb-0f55-314c878ba3ec
# ╠═95a667de-880d-11eb-0171-b93ed1f38ea1
# ╟─633b3d12-76a4-11eb-0bc7-b9bf9116933f
# ╟─4b9f5030-76cc-11eb-117c-91ca8336c30b
# ╠═023173fe-8644-11eb-3303-e351dbf44aaf
# ╟─7278ba0a-8b00-11eb-3629-e55ab965940c
# ╟─3402cd3c-8afc-11eb-2af1-312ae538cd1a
# ╟─82ed33a0-8b00-11eb-11d0-cddce2e38e2c
# ╟─990dd67c-8afc-11eb-0f5d-f1525f921906
# ╠═50bc7ea0-8afc-11eb-1101-d7a7373ed0ce
# ╠═5f7bbdc0-8afc-11eb-1b38-e3448733ad4b
# ╟─81f270e2-8afc-11eb-24be-5952f95e6aa3
# ╟─cbb19904-8afc-11eb-19a5-47ad0bae2bfd
# ╟─b1a3c0a6-8643-11eb-1a7b-cd4720e77617
# ╠═fa52bcd0-76f8-11eb-0d58-955a514a00b1
# ╠═4e66a016-76f9-11eb-2023-6dfc3374c066
# ╟─106d3bc0-76fa-11eb-1ee6-3fa73be52226
# ╟─e2cbc0ec-76f9-11eb-2870-f10f6cdc8be4
# ╟─9449905c-8b15-11eb-0987-471b19ff966b
# ╟─a186f7f2-8b15-11eb-195d-5fe71ec9fd1e
# ╟─46e7c83a-8bdd-11eb-211d-31a8a2b766c3
# ╠═435e9954-8b16-11eb-06fa-f70df37efee9
# ╟─5125d26e-8b16-11eb-2da0-235368e7840c
# ╠═6e46e702-8b16-11eb-2edf-e12f7c97594d
# ╟─77e8e9ce-8bd0-11eb-0d60-cf66bd337f0c
# ╟─3ab28264-6c64-11eb-29f4-a9ed2e9eba16
# ╟─d32173ec-66e8-11eb-11ad-f9605b4964b2
