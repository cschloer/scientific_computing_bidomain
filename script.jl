using ExtendableGrids

function create_grid(n,dim)
	nx=n
	if dim==2
		nx=ceil(sqrt(n))
	end
	X=collect(0:70/nx:70)
	if dim==1
      grid=simplexgrid(X)
	else
      grid=simplexgrid(X,X)
	end
	return grid,X
end

g,X = create_grid(100, 2)

print(g)
print("\n")
print(X)
print("\n")
print(g[1])
print("\n")
print("\n")
print(num_nodes(g))
