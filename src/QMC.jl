"""	
```julia
compute_with_QMC(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)
```

This function computes a separable joint probability function and its gradient using 
Quasi-Monte Carlo. It returns the following function at a first position:

```math
f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ),
```

In a second position, it returns the gradient of the above function, which is computed using the Prékopa theorem as explained in Section X.

The function takes 6 arguments:
* `j` - index in Σ and μ from which the joint probability applies
* `κ` - dimension of the multivariate distribution minus 1
* `Σ` - covariance matrix
* `μ` - mean vector
* `s` - sample
* `rng` - random number generator

If `s` and `rng` are not specified, they take the default values `s = 5000` and `rng = MersenneTwister(1234)`.
With the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.
This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability 
function is used in an optimization problem. 
If the goal is only the numerical computation of the probability function without further integration into an optimization
problem, the argument `rng` can be unfixed by passing the value `RandomDevice()`.

"""
function compute_with_QMC(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer = 5000, rng = MersenneTwister(1234))
	
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
				g[i] = 0.00000
			end
		end
		return
	end
	# TODO add note that after computing f and ∇f to assert errors are not 0.0
	return f, ∇f, e 
end

"""
```julia
add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
```

This function allows to add a system of joint chance constraints into a JuMP model as follows:

```math
∀ \\quad j \\text{ in } \\quad idx:

f(x) = \\mathbb{P}(g_i (x, ξ) ≥ 0 \\quad ∀i = j, ..., j+κ) ≥ p
```

The function takes 6 arguments:
	* `m` - a JuMP model
	* `x` - a vector of decision variables
	* `idx` - set of indices that constitue the starting time of a reliability window
	* `κ` - dimension of the multivariate distribution minus 1
	* `Σ` - covariance matrix
	* `μ` - mean vector
	* `p` - probability level that has to be met
	
To add this system of nonlinear constraints to the model, the function defines the probability function 
computed using the quasi-monte-carlo method in `compute_with_QMC` as a user-defined operator 
and provides it also with the gradient, also computed using `compute_with_QMC.

"""
function add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)
    for j in idx
		# TODO update to non-legacy
        JuMP.register(m, Symbol("mvncdf_$j"), length(x), compute_with_QMC(j, κ, Σ, μ, 5000, MersenneTwister(1234))[1], compute_with_QMC(j, κ, Σ, μ, 5000, MersenneTwister(1234))[2])
        JuMP.add_nonlinear_constraint(m, :($(Symbol("mvncdf_$j"))($(x...)) >= $(p)))
    end
end