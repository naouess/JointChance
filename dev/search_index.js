var documenterSearchIndex = {"docs":
[{"location":"jcc/#Joint-probability-functions","page":"Joint Chance Contraints","title":"Joint probability functions","text":"","category":"section"},{"location":"jcc/#Computing-probability-functions","page":"Joint Chance Contraints","title":"Computing probability functions","text":"","category":"section"},{"location":"jcc/","page":"Joint Chance Contraints","title":"Joint Chance Contraints","text":"compute_with_QMC","category":"page"},{"location":"jcc/#JointChance.compute_with_QMC","page":"Joint Chance Contraints","title":"JointChance.compute_with_QMC","text":"compute_with_QMC(j::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)\n\nThis function computes a separable joint probability function and its gradient using  Quasi-Monte Carlo. It returns the following function at a first position:\n\nf(x) = mathbbP(g_i (x ξ)  0 quad i = j  j+κ)\n\nIn a second position, it returns the gradient of the above function, which is computed using the Prékopa theorem as explained in Section X.\n\nThe function takes 6 arguments:\n\nj - index in Σ and μ from which the joint probability applies\nκ - dimension of the multivariate distribution minus 1\nΣ - covariance matrix\nμ - mean vector\ns - sample\nrng - random number generator\n\nIf s and rng are not specified, they take the default values s = 5000 and rng = MersenneTwister(1234). With the latter, we fix the random number generator rng to guarantee the stability of the gradients in the solver iterations. This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability  function is used in an optimization problem.  If the goal is only the numerical computation of the probability function without further integration into an optimization problem, the argument rng can be unfixed by passing the value RandomDevice().\n\n\n\n\n\n","category":"function"},{"location":"jcc/","page":"Joint Chance Contraints","title":"Joint Chance Contraints","text":"compute_with_SRD","category":"page"},{"location":"jcc/#JointChance.compute_with_SRD","page":"Joint Chance Contraints","title":"JointChance.compute_with_SRD","text":"compute_with_SRD(w::Integer, κ::Integer, Σ::AbstractMatrix, μ::AbstractVector, s::Integer, rng)\n\nThis function computes a separable joint probability function and its gradient using  Spherical Radial Decomposition. It returns an array consisting of the following function in its first element:\n\nf(x) = mathbbP(g_i (x ξ)  0 quad i = j  j+κ)\n\nAs a second element, it returns the gradient of the above function, which is computed using the Prékopa theorem as explained in Section X.\n\nThe function takes 6 arguments:\n\nw - index in Σ and μ from which the joint probability applies\nκ - dimension of the multivariate distribution minus 1\nΣ - covariance matrix\nμ - mean vector\ns - sample\nrng - random number generator\n\nIf s and rng are not specified, they take the default values s = 5000 and rng = MersenneTwister(1234). With the latter, we fix the random number generator rng to guarantee the stability of the gradients in the solver iterations. This tempers with the precision of the computation, yet it is necessary for solver convergence, in case the probability  function is used in an optimization problem.  If the goal is only the numerical computation of the probability function without further integration into an optimization problem, the argument rng can be unfixed by passing the value RandomDevice()\n\n\n\n\n\n","category":"function"},{"location":"jcc/#Adding-chance-constraints-to-a-JuMP-model","page":"Joint Chance Contraints","title":"Adding chance constraints to a JuMP model","text":"","category":"section"},{"location":"jcc/","page":"Joint Chance Contraints","title":"Joint Chance Contraints","text":"add_JCC_QMC","category":"page"},{"location":"jcc/#JointChance.add_JCC_QMC","page":"Joint Chance Contraints","title":"JointChance.add_JCC_QMC","text":"add_JCC_QMC(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)\n\nThis function allows to add a system of joint chance constraints into a JuMP model as follows:\n\n quad j text in  quad idx\n\nf(x) = mathbbP(g_i (x ξ)  0 quad i = j  j+κ)  p\n\nThe function takes 6 arguments: \t* m - a JuMP model \t* x - a vector of decision variables \t* idx - set of indices that constitue the starting time of a reliability window \t* κ - dimension of the multivariate distribution minus 1 \t* Σ - covariance matrix \t* μ - mean vector \t* p - probability level that has to be met\n\nTo add this system of nonlinear constraints to the model, the function defines the probability function  computed using the quasi-monte-carlo method in compute_with_QMC as a user-defined operator  and provides it also with the gradient, also computed using `computewithQMC.\n\n\n\n\n\n","category":"function"},{"location":"jcc/","page":"Joint Chance Contraints","title":"Joint Chance Contraints","text":"add_JCC_SRD","category":"page"},{"location":"jcc/#JointChance.add_JCC_SRD","page":"Joint Chance Contraints","title":"JointChance.add_JCC_SRD","text":"add_JCC_SRD(m::JuMP.Model, x::AbstractVector, idx::AbstractArray, κ::Integer, Σ::AbstractMatrix, μ::AbstractArray, p::Float64)\n\nThis function allows to add a system of joint chance constraints into a JuMP model as follows:\n\n\n quad j text in  quad idx\n\nf(x) = mathbbP(g_i (x ξ)  0 quad i = j  j+κ)  p\n\nThe function takes 6 arguments:\n\nm - a JuMP model\nx - a vector of decision variables\nidx - set of indices that constitue the starting time of a reliability window\nκ - dimension of the multivariate distribution minus 1\nΣ - covariance matrix\nμ - mean vector\np - probability level that has to be met\n\nTo add this system of nonlinear constraints to the model, the function defines the probability function  computed using the spherical radial decomposition method in compute_with_SRD as a user-defined operator  and provides it also with the gradient, also computed using `computewithSRD.\n\n\n\n\n\n","category":"function"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#First-example","page":"Examples","title":"First example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Newsvendor with Individual Chance Constraint \nNewsvendor with Joint Chance Constraint","category":"page"},{"location":"examples/#Second-example","page":"Examples","title":"Second example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Financial portfolio ICC \nFinancial portfolio JCC","category":"page"},{"location":"examples/#Third-Example","page":"Examples","title":"Third Example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Hydropower Dispatch ICC \nHydropower Dispatch JCC","category":"page"},{"location":"icc/#Individual-probability-functions","page":"Individual Chance Contraints","title":"Individual probability functions","text":"","category":"section"},{"location":"#JointChance.jl","page":"Home","title":"JointChance.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A package for dealing with Joint Chance Constraints in Julia.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\"]","category":"page"},{"location":"#Install","page":"Home","title":"Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"# pkg> add \"https://github.com/naouess/JointChance\"","category":"page"},{"location":"#Resources-for-getting-started","page":"Home","title":"Resources for getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Here some math stuff \nSome JuMP and Julia stuff \nRefer to our paper in Annals of OR","category":"page"},{"location":"#The-math-behind-it-all","page":"Home","title":"The math behind it all","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Here we explain Chance constraints ","category":"page"}]
}
