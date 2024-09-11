module JointChance
    using JuMP, MvNormalCDF, Random, Distributions, SpecialFunctions, LinearAlgebra

    export add_JCC_SRD, add_JCC_Genz, compute_with_SRD, compute_with_Genz

    include("chi_gamma.jl")
    include("SRD.jl")
    include("Genz.jl")
end

