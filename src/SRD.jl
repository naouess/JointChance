#=
    ProbFunction(x, SampleOnSphere, mu)

This function computes the probability of a given vector `x` 
being less than or equal to a normal random variable `xi` 
with mean `mu` and a given covariance matrix, based on a sample
of points distributed on the unit sphere.
=#
function ProbFunction(x, SampleOnSphere, mu)
      
    SampleSize, M = size(SampleOnSphere)
    PROB = 0
    
    # Required gamma values for Chi-distibution
    gm = [loggamma(M/2.0), loggamma(M/2.0 + 1.0), gamma(M/2.0)]
    
    # setup gradient
    grad = zeros(size(x, 1), 1)
    
    for sample = 1:SampleSize
        
        # Select sample on sphere
        v = SampleOnSphere[sample, :]'
        
        # Compute radius r (Inner problem)
        
        # Pre-Initialization
        r = 1e99
        active = 0
        
        y = x - mu
        
        for k = 1:M
            
            # k-th component
            rstep = y[k] / v[k]
            
            if rstep > 0
                if rstep < r
                    r = rstep
                    active = k
                end
            end
            
        end
        
        # Compute chi-distribution of r
        
        # PROB
        if r > 0
            chi = chi_cdf_gamma(r, M, gm)
            PROB += chi
        end
        
        # Gradient of ProbFunction
          
        if r > 0
            
            if active > 0
                
                # active inequality for r
                gradz = zeros(M, 1)
                gradz[active] = 1
                factor = dot(gradz', v)
                factor = chi_pdf_gamma(r, M, gm) / factor
                
                # update grad_vector components
                gradx = zeros(M, 1)
                gradx[active] = -1
                grad -= factor .* gradx
                
            end
            
        end
              
    end
    
    PROB = PROB / SampleSize
    
    # Value of constraint:  probLevel - Phi(u) <= 0
    prob = PROB
    
    # Gradient of constraint: - Grad(Phi(u))
    grad = grad / SampleSize
        
    return (prob, grad)
end

"""	
```julia
compute_with_SRD(j, κ, Σ, μ, size, rng)
```

This function computes a separable joint probability function and its gradient using 
Spherical Radial Decomposition. The function is defined as

```math
f(x) = h(x(t) ≥ ξ(t) ∀t = j, ..., j+κ).
```

The function takes 6 arguments:
* `w` - index in Σ and μ from which the joint probability applies.
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
function compute_with_SRD(w, κ, Σ, μ, size, rng)
    # TODO add checks about validity of inputs

    # Set dimension of each probability constraint
    dim_x = length(μ)
    dim_cons = κ + 1
    SampleSize = size

    # Compute cholesky of covariance
    L = cholesky(Σ, check=true).L

    # Compute Sample on the sphere
    SampleOnSphere = randn(rng, SampleSize, dim_cons*(dim_x-dim_cons+1))

    for j in 1:dim_x-dim_cons+1
        i = dim_cons * (j-1) + 1
        for n = 1:SampleSize
        # normalize sample (projection to sphere)
            v = SampleOnSphere[n, i:i+dim_cons-1]'
            v = v / norm(v)

            # transformation with cholesky of covariance
            v = L[j:j+dim_cons-1, j:j+dim_cons-1] * v'

            # save result
            SampleOnSphere[n, i:i+dim_cons-1] = v'
        end 
    end

    k = (1+κ) * (w-1) + 1

    f(x...) = ProbFunction(vec([x[r] for r in w:w+κ]), SampleOnSphere[:, k:k+κ], μ[w:w+κ])[1]

    function ∇f(g::AbstractVector{T}, x::T...) where {T}
        for i in eachindex(x)
            if i in w:w+κ
                g[i] = ProbFunction(vec([x[r] for r in w:w+κ]), SampleOnSphere[:, k:k+κ], μ[w:w+κ])[2][i-w+1]
            else 
                g[i] = 0.000
            end 
        end
        return
    end
    # TODO assert results
    return f, ∇f
end 

function compute_with_SRD(w, κ, Σ, μ; size = 5000, rng = MersenneTwister(1234))
    compute_with_SRD(w, κ, Σ, μ, size = size, rng = rng)
end

"""
```julia
add_JCC_SRD(m, x, idx, κ, Σ, μ, p)
```

"""
function add_JCC_SRD(m, x, idx, κ, Σ, μ, p)
    for j in idx
        JuMP.add_nonlinear_operator(m, Symbol("srd_prob_$j"), length(x), compute_with_SRD(j, κ, Σ, μ, 5000, MersenneTwister(1234))[1], compute_with_SRD(j, κ, Σ, μ, 5000, MersenneTwister(1234))[2])
        JuMP.add_nonlinear_constraint(m, :($(Symbol("srd_prob_$j"))($(x...)) >= $(p)))
    end
end

function add_JCC_SRD(m, x, idx, κ, dist, p)
    add_JCC_SRD(m, x, idx, κ, dist.Σ, dist.μ, p)
end