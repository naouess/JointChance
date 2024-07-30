module JointChance
    using JuMP, MvNormalCDF, Random, Distributions, SpecialFunctions, LinearAlgebra

    export add_constraints_SRD, add_constraints_Genz, compute_with_SRD, compute_with_Genz

    include("chi_gamma.jl")
    include("SRD.jl")
    include("Genz.jl")
end

