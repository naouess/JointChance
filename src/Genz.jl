"""	
```julia
compute_with_Genz(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)
```

This function computes a separable joint probability function and its gradient using 
Quasi-Monte Carlo. It returns the following function at a first position:

```math
f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ),
```

In a second position, it returns the gradient of the above function, defined using the Prékopa 
theorem explained in Section X.

The function takes 6 arguments:
* `j` - index in Σ and μ from which the joint probability applies.
* `κ` - duration of an outage
* `Σ` - covariance matrix
* `μ` - mean vector
* `s` - sample
* `rng` - random number generator

If `s` and `rng` are not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can take the values `RandomDevice()` #TODO

"""
function compute_with_Genz(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)
	
	# assert validity of inputs 
	@assert all(v -> v > 0, (j, κ, s)) "The parameters j, κ, s should be strictly positive."
	@assert size(Σ) == (length(μ), length(μ)) "The covariance matrix should be symetrical and have the same dimension as the mean vector μ."

	a = vec([-Inf for i in eachindex(μ)])
	norm_pdf(k) = exp(-(k^2) / 2) / (2 * pi)^0.5
	
	f(x...) = mvnormcdf(
					Σ[j:j+κ, j:j+κ], 
					a[j:j+κ]-μ[j:j+κ], 
					vec([x[i] for i in j:j+κ])-μ[j:j+κ], 
	                m = s, 
					rng = rng)[1]
	
	# define the error
	e(x...) = mvnormcdf(
						Σ[j:j+κ, j:j+κ], 
						a[j:j+κ]-μ[j:j+κ], 
						vec([x[i] for i in j:j+κ])-μ[j:j+κ], 
						m = s, 
						rng = rng)[2]

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
                            		m = s, 
									rng = rng)[1]  
			else 
				g[i] = 0.000
			end
		end
		return
	end
	# TODO add note that after computing f and ∇f, please assert errors are not 0.0?
	return f, ∇f, e # TODO add ∇e
end

# set default values for m and rng (fixed and large sample)
function compute_with_Genz(j, κ, Σ, μ; s = 5000, rng = MersenneTwister(1234)) 
	compute_with_Genz(j, κ, Σ, μ, s = s, rng = rng)
end 

"""
```julia
add_JCC_Genz(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
```

This function allows to add joint chance constraints into a JuMP model. 
To do so, it defines the probability function computed using the quasi-monte-carlo method defined in `compute_with_Genz` as 
a user-defined operator and provides it also with the gradient, also computed using `compute_with_Genz.
"""
function add_JCC_Genz(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
    for j in idx
        JuMP.add_nonlinear_operator(m, Symbol("mvncdf_$j"), length(x), compute_with_Genz(j, κ, Σ, μ, 5000, MersenneTwister(1234))[1], compute_with_Genz(j, κ, Σ, μ, 5000, MersenneTwister(1234))[2])
        JuMP.add_nonlinear_constraint(m, :($(Symbol("mvncdf_$j"))($(x...)) >= $(p)))
    end
end

#=
TODO
=#
function add_JCC_Genz(m, x, idx, κ, dist, p)
    add_JCC_Genz(m, x, idx, κ, dist.Σ, dist.μ, p)
end 
