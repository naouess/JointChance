var documenterSearchIndex = {"docs":
[{"location":"functions/#Functions","page":"Functions","title":"Functions","text":"","category":"section"},{"location":"functions/#Computing-a-probability-function-and-its-gradient","page":"Functions","title":"Computing a probability function and its gradient","text":"","category":"section"},{"location":"functions/","page":"Functions","title":"Functions","text":"compute_with_Genz","category":"page"},{"location":"functions/#JointChance.compute_with_Genz","page":"Functions","title":"JointChance.compute_with_Genz","text":"```julia\ncompute_with_Genz(j, κ, Σ, μ, size, rng)\n```\nThis function computes a separable joint probability function and its gradient using \nQuasi-Monte Carlo. The function is defined as\n\n```math\nf(x) = h(x(t) ≥ ξ(t) ∀t = j, ..., j+κ).\n```\n\nThe function takes 6 arguments:\n* `j` - index in Σ and μ from which the joint probability applies.\n* `κ` - duration of an outage\n* `Σ` - covariance matrix\n* `μ` - mean vector\n* `size` - sample\n* `rng` - random number generator\n\nIf `size` and `rng` are not specified, they take the default values `size = 5000` and `rng = MersenneTwister(1234)`.\nWith the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.\nThis tempers with the precision of the solution found, yet it is necessary for solver convergence.\n# TODO formulate better here\n\n\n\n\n\n","category":"function"},{"location":"functions/","page":"Functions","title":"Functions","text":"compute_with_SRD","category":"page"},{"location":"functions/#JointChance.compute_with_SRD","page":"Functions","title":"JointChance.compute_with_SRD","text":"```julia\ncompute_with_SRD(j, κ, Σ, μ, size, rng)\n```\nThis function computes a separable joint probability function and its gradient using \nSpherical Radial Decomposition. The function is defined as\n\n```math\nf(x) = h(x(t) ≥ ξ(t) ∀t = j, ..., j+κ).\n```\n\nThe function takes 6 arguments:\n* `w` - index in Σ and μ from which the joint probability applies.\n* `κ` - duration of an outage\n* `Σ` - covariance matrix\n* `μ` - mean vector\n* `size` - sample\n* `rng` - random number generator\n\nIf `size` and `rng` are not specified, they take the default values `size = 5000` and `rng = MersenneTwister(1234)`.\nWith the latter, we fix the random number generator `rng` to guarantee the stability of the gradients in the solver iterations.\nThis tempers with the precision of the solution found, yet it is necessary for solver convergence.\n# TODO formulate better here\n\n\n\n\n\n","category":"function"},{"location":"functions/#Adding-the-probablity-functions-as-constraints-to-an-existing-JuMP-model","page":"Functions","title":"Adding the probablity functions as constraints to an existing JuMP model","text":"","category":"section"},{"location":"functions/","page":"Functions","title":"Functions","text":"add_constraints_Genz","category":"page"},{"location":"functions/#JointChance.add_constraints_Genz","page":"Functions","title":"JointChance.add_constraints_Genz","text":"\tadd_constraints_Genz(m, x, idx, κ, Σ, μ, p)\n\n\n\n\n\n","category":"function"},{"location":"functions/","page":"Functions","title":"Functions","text":"add_constraints_SRD","category":"page"},{"location":"functions/#JointChance.add_constraints_SRD","page":"Functions","title":"JointChance.add_constraints_SRD","text":"\tadd_constraints_SRD(m, x, idx, κ, Σ, μ, p)\n\n\n\n\n\n","category":"function"},{"location":"examples/#Examples","page":"Examples","title":"Examples","text":"","category":"section"},{"location":"examples/#First-example","page":"Examples","title":"First example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Newsvendor with Individual Chance Constraint Newsvendor with Joint Chance Constraint","category":"page"},{"location":"examples/#Second-example","page":"Examples","title":"Second example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Financial portfolio ICC Financial portfolio JCC","category":"page"},{"location":"examples/#Third-Example","page":"Examples","title":"Third Example","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"Hydropower Dispatch ICC Hydropower Dispatch JCC","category":"page"},{"location":"#JointChance.jl","page":"Home","title":"JointChance.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A package for dealing with Joint Chance Constraints in Julia.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"index.md\"]","category":"page"},{"location":"#Install","page":"Home","title":"Install","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"# pkg> add \"https://github.com/naouess/JointChance\"","category":"page"},{"location":"#Resources-for-getting-started","page":"Home","title":"Resources for getting started","text":"","category":"section"},{"location":"#The-Math-behind-it-all","page":"Home","title":"The Math behind it all","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Here we explain Chance constraints ","category":"page"}]
}
