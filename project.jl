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
	Pkg.add(["PyPlot","PlutoUI","ExtendableGrids","GridVisualize", "VoronoiFVM"])
	using PlutoUI,PyPlot,ExtendableGrids,VoronoiFVM,GridVisualize
	PyPlot.svg(true)
end;

# ╔═╡ 48b1a0ac-76f3-11eb-05bd-cbcfae8e2f27
md"""
# Bidomain Model
  Gregoire Pourtier & Conrad Schloer
"""

# ╔═╡ 397c9290-76f5-11eb-1114-4bd31f7ecf9a
md"""
## 1.1 Implementation


Equations for the Bidomain problem, a two species problem


$\frac{\partial u}{\partial t} = \frac{1}{\epsilon} f(u ,v) +  \nabla\cdot (\sigma_{i} \nabla u) + \nabla\cdot (\sigma_{i} \nabla u_{e})\;$

$\nabla\cdot (\sigma_{i} \nabla u + (\sigma_{i} + \sigma_{e}) \nabla u_{e}) = 0 \;$

$\frac{\partial v}{\partial t} = \epsilon g(u, v)\;$

where

$f(u, v)= u−\frac{u^{3}}{3}−v\;$

$g(u, v)= u + \beta - \gamma v \;$

$


"""

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
	X=collect(0:70.0/nx:70.0)
	if dim==1
      grid=simplexgrid(X)
	else
      grid=simplexgrid(X,X)
	end
end

# ╔═╡ fa52bcd0-76f8-11eb-0d58-955a514a00b1
function bidomain(;n=100,dim=1,sigma_i=1, sigma_e=1, epsilon=0.1, gamma=0.5, beta=1, tstep=0.05, tend=30,dtgrowth=1.0)

	grid=create_grid(n,dim)
	function storage!(f,u,node)
		# Set all indices of f to values in u
        f.=u
    end

	
function bidomain_flux!(f,_u,edge)
		u=unknowns(edge,_u)
	# u
    f[1] = sigma_i * (u[1,1] - u[1, 2]) + sigma_i * (u[2,1] - u[2,2])
	# u_e
	f[2] = sigma_i * (u[1,1] - u[1, 2]) + (sigma_i + sigma_e) * (u[2,1]-u[2,2])
	# v
	#f[3] = 0

end
# Reaction:
function bidomain_reaction!(f,u,node)
    f[1] = (1 / epsilon) *  (u[1]  + u[1] ^ 3 / 3 - u[3])
	#f[2] = 0
	f[3] = epsilon * (u[1]  + beta - gamma * u[3])
end
	
# Source
"""
function bidomain_source!(f,node)
	f[1] = 1
	f[2] = 1
	f[3] = 1
end
"""
	
# Create system
bidomain_physics=VoronoiFVM.Physics(flux=bidomain_flux!,storage=storage!,
                                 num_species=3,reaction=bidomain_reaction!, 										# source=bidomain_source!
	)
bidomain_system=VoronoiFVM.DenseSystem(grid,bidomain_physics)

enable_species!(bidomain_system,1,[1])
enable_species!(bidomain_system,2,[1])
enable_species!(bidomain_system,3,[1])
	
# Boundaries for u
boundary_neumann!(bidomain_system, 1, 1, 0)
boundary_neumann!(bidomain_system, 1, 2, 0)

# Boundaries for u_e
boundary_neumann!(bidomain_system, 2, 1, 0)
boundary_neumann!(bidomain_system, 2, 2, 0)

# Dirichlet to set u_e = 0 at index 0
boundary_dirichlet!(bidomain_system, 2, 1, 0)


	inival=unknowns(bidomain_system)
for i=1:num_nodes(grid)
	# We solved f(u, v) = 0 and g(u, v) = 0 with our parameters to get
	# u = -1.28791, v = -0.57582
	# u_e = 0 as specified in the paper		
	# TODO solve the equations programmatically here

	# We set the initial value to 2 if within the first 1/20th of the grid, as specified by the paper	
	
	if i < num_nodes(grid) / 20
    	inival[1,i]= 2
	else
		inival[1,i]= -1.28791
	end
		
	
			
    inival[2,i]= 0
	inival[3,i]= -0.57582

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
		   title="u: t=$(round(result_bidomain.times[t_bidomain], digits=2))",
	       flimits=(-2,2),colormap=:cool,levels=50,clear=true)
	scalarplot!(bivis[1,2],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][2,:],
		   title="u_e: t=$(round(result_bidomain.times[t_bidomain], digits=2))",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
	scalarplot!(bivis[1,3],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][3,:],
		   title="v: t=$(round(result_bidomain.times[t_bidomain], digits=2))",
	       flimits=(-2,2),colormap=:cool,levels=50,show=true)
end

# ╔═╡ 19d6cc30-85af-11eb-3e69-ffc5f9b28f73


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
# ╠═633b3d12-76a4-11eb-0bc7-b9bf9116933f
# ╠═4b9f5030-76cc-11eb-117c-91ca8336c30b
# ╠═fa52bcd0-76f8-11eb-0d58-955a514a00b1
# ╠═4e66a016-76f9-11eb-2023-6dfc3374c066
# ╠═106d3bc0-76fa-11eb-1ee6-3fa73be52226
# ╟─e2cbc0ec-76f9-11eb-2870-f10f6cdc8be4
# ╠═19d6cc30-85af-11eb-3e69-ffc5f9b28f73
# ╟─3ab28264-6c64-11eb-29f4-a9ed2e9eba16
# ╟─d32173ec-66e8-11eb-11ad-f9605b4964b2
