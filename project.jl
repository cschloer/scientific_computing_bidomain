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


# ╔═╡ a7724d46-76cb-11eb-2275-0b6db01e36b2
# Define function for initial value $u_0$ with two methods - for 1D and 2D problems
begin
	fpeak(x)=exp(-100*(x-0.25)^2)
	fpeak(x,y)=exp(-100*((x-0.25)^2+(y-0.25)^2))
end

# ╔═╡ 4b9f5030-76cc-11eb-117c-91ca8336c30b
# Create discretization grid in 1D or 2D with approximately n nodes
function create_grid(n,dim)
	nx=n
	if dim==2
		nx=ceil(sqrt(n))
	end
	X=collect(0:1.0/nx:1)
	if dim==1
      grid=simplexgrid(X)
	else
      grid=simplexgrid(X,X)
	end
end

# ╔═╡ fa52bcd0-76f8-11eb-0d58-955a514a00b1
function bidomain(;n=100,dim=1,A=4.0,B=6.0,D1=0.01,D2=0.1,perturbation=0.1,
	tstep=0.05, tend=150,dtgrowth=1.05)

	grid=create_grid(n,dim)
	function storage!(f,u,node)
        f.=u
    end

	
function bidomain_diffusion!(f,_u,edge)
		u=unknowns(edge,_u)
    f[1]=D1*(u[1,1]-u[1,2])
    f[2]=D2*(u[2,1]-u[2,2])
end
# Reaction:
function bidomain_reaction!(f,u,node)
    f[1]= (B+1.0)*u[1]-A-u[1]^2*u[2]
    f[2]= u[1]^2*u[2]-B*u[1]
end
# Create system
bidomain_physics=VoronoiFVM.Physics(flux=bidomain_diffusion!,storage=storage!,
                                 num_species=2,reaction=bidomain_reaction!)
bidomain_system=VoronoiFVM.DenseSystem(grid,bidomain_physics)
enable_species!(bidomain_system,1,[1])
enable_species!(bidomain_system,2,[1])

	inival=unknowns(bidomain_system)
for i=1:num_nodes(grid)
    inival[1,i]=1.0+perturbation*randn()
    inival[2,i]=1.0+perturbation*randn()
end

	evolution(inival,bidomain_system,grid,tstep,tend,dtgrowth)	
end


# ╔═╡ 4e66a016-76f9-11eb-2023-6dfc3374c066
result_bidomain=bidomain(n=50,dim=1);

# ╔═╡ 106d3bc0-76fa-11eb-1ee6-3fa73be52226
md"""
time=$(@bind t_bidomain Slider(1:length(result_bidomain.times),default=1))
"""

# ╔═╡ e2cbc0ec-76f9-11eb-2870-f10f6cdc8be4
let
	bivis=GridVisualizer(layout=(1,2),resolution=(600,300),Plotter=PyPlot)
	scalarplot!(bivis[1,1],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][1,:],
		   title="u1: t=$(result_bidomain.times[t_bidomain])",
	       flimits=(0,10),colormap=:cool,levels=50,clear=true)
	scalarplot!(bivis[1,2],result_bidomain.grid,
	       result_bidomain.solutions[t_bidomain][2,:],
		   title="u2: t=$(result_bidomain.times[t_bidomain])",
	       flimits=(0.5,3),colormap=:cool,levels=50,show=true)
end

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
# ╠═a7724d46-76cb-11eb-2275-0b6db01e36b2
# ╠═4b9f5030-76cc-11eb-117c-91ca8336c30b
# ╠═fa52bcd0-76f8-11eb-0d58-955a514a00b1
# ╠═4e66a016-76f9-11eb-2023-6dfc3374c066
# ╠═106d3bc0-76fa-11eb-1ee6-3fa73be52226
# ╠═e2cbc0ec-76f9-11eb-2870-f10f6cdc8be4
# ╟─3ab28264-6c64-11eb-29f4-a9ed2e9eba16
# ╟─d32173ec-66e8-11eb-11ad-f9605b4964b2
