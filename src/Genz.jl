"""	
	```julia
	compute_with_Genz(j, κ, Σ, μ, size, rng)
	```
	This function computes a separable joint probability function and its gradient using 
	Quasi-Monte Carlo. The function is defined as
	
	```math
	f(x) = h(x(t) ≥ ξ(t) ∀t = j, ..., j+κ).
	```

	The function takes 6 arguments:
	* `j` - index in Σ and μ from which the joint probability applies.
	* `κ` - duration of an outage
	* `Σ` - covariance matrix
	* `μ` - mean vector
	* `size` - sample
	* `rng` - random number generator

	If `size` and `rng` are not specified, they take the default values `size = 5000` and `rng = MersenneTwister(1234)`.
	With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
	This tempers with the precision of the solution found, yet it is necessary for solver convergence.
	# TODO formulate better here

"""
function compute_with_Genz(j, κ, Σ, μ, size, rng)
	# TODO add checks about validity of inputs
	a = vec([-Inf for i in eachindex(μ)])
	norm_pdf(k) = exp(-(k^2) / 2) / (2 * pi)^0.5
	f(x...) = mvnormcdf(
					Σ[j:j+κ, j:j+κ], 
					a[j:j+κ]-μ[j:j+κ], 
					vec([x[i] for i in j:j+κ])-μ[j:j+κ], 
	                m = size, 
					rng = rng)[1]

	# TODO are mu, sigma, Sigma and a global variables for the function ∇f? accessible to it with no confusion?
    function ∇f(g::AbstractVector{T}, x::T...) where {T}
		μ_j = μ[j:j+κ]
		Σ_j = Σ[j:j+κ, j:j+κ]
		σ_j = diag(Σ_j, 0).^0.5
		a_j = a[j:j+κ]
		for i in eachindex(x)
			if i in j:j+κ
				Σ_new = Σ_j - inv(Σ_j[i-j+1, i-j+1]) * Σ_j[i-j+1, :] * transpose(Σ_j[i-j+1,:])
				μ_new = μ_j + inv(Σ_j[i-j+1, i-j+1]) * (x[i] - μ_j[i-j+1]) * Σ_j[i-j+1, :]
				g[i] = norm_pdf((x[i] - μ_j[i-j+1])/σ_j[i-j+1]) / σ_j[i-j+1] * 
								 mvnormcdf(
									Σ_new[1:end .!= i-j+1, 1:end .!= i-j+1], 
                                 	a_j[1:end .!=i-j+1]-μ_new[1:end .!=i-j+1], 
									vec([x[k] for k in j:j+κ if k != i])-μ_new[1:end .!=i-j+1], 
                            		m = size, 
									rng = rng)[1]  
			else 
				g[i] = 0.000
			end
		end
		return
	end
    # TODO assert errors are not 0.0
	return f, ∇f
end

# TODO set default values for m and rng (fixed and large sample) fix this
function compute_with_Genz(j, κ, Σ, μ; size = 5000, rng = MersenneTwister(1234)) 
	compute_with_Genz(j, κ, Σ, μ, size = size, rng = rng)
end 

"""
```julia
	add_constraints_Genz(m, x, idx, κ, Σ, μ, p)
```
"""
function add_constraints_Genz(m, x, idx, κ, Σ, μ, p) # Constraint_JointProb_Genz() & Constraint_JointProb_SRD
    for j in idx
        JuMP.register(m, Symbol("mvncdf_$j"), length(x), compute_with_Genz(j, κ, Σ, μ, 5000, MersenneTwister(1234))[1], compute_with_Genz(j, κ, Σ, μ, 5000, MersenneTwister(1234))[2])
        JuMP.add_nonlinear_constraint(m, :($(Symbol("mvncdf_$j"))($(x...)) >= $(p)))
    end
end

function add_constraints_Genz(m, x, idx, κ, dist, p)
    add_constraints_Genz(m, x, idx, κ, dist.Σ, dist.μ, p)
end 
