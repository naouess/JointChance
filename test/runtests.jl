using JointChance
using Test
using Random

p = 0.9
μ = [5; 2; 4; 3;]
Σ = [
    2.0   0.5   0.3  -0.1;
    0.5   2.2   0.7   0.4;
    0.3   0.7   1.6  -0.2;
   -0.1   0.4  -0.2   1.8;
]

κ = 3

x = [8; 6; 7; 5;]

prob_qmc  = compute_with_QMC(1, κ, Σ, μ, 10000, MersenneTwister(1234))[1](x...)
prob_srd = compute_with_SRD(1, κ, Σ, μ, 10000, MersenneTwister(1234))[1](x...)

v1 = round.(prob_qmc, digits=6)
v2 = round.(prob_srd, digits=6)

prob_qmc1 = compute_with_QMC(1, κ, Σ, μ)[1](x...)
prob_srd1 = compute_with_SRD(1, κ, Σ, μ)[1](x...)

v3 = round.(prob_qmc1, digits=6)
v4 = round.(prob_srd1, digits=6)

prob_qmc_default = zeros()
prob_srd_default = zeros()
for j in [1, 2]
    prob_qmc_default = [prob_qmc_default[j] compute_with_QMC(j, κ-1, Σ, μ)[1](x...)]
    prob_srd_default = [prob_srd_default[j] compute_with_SRD(j, κ-1, Σ, μ)[1](x...)]
end

v5 = round.(prob_qmc_default, digits=6)
v6 = round.(prob_srd_default, digits=6)

@testset "JointChance.jl" begin
    @test v1 == 0.905384
    @test v2 == 0.903887
    @test v3 == 0.905395
    @test v4 == 0.903956
    @test v5 == [0.971658  0.920864]
    @test v6 == [0.971249  0.922053]
end

# TODO add test for the functions adding constraints to a JuMP model