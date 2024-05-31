using .JointChance
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

prob  = compute_with_genz(1, κ, Σ, μ, 10000, MersenneTwister(1234))[1](x...)
prob1 = compute_with_SRD(1, κ, Σ, μ, 10000, MersenneTwister(1234))[1](x...)

v = round.(prob, digits=6)
v1 = round.(prob1, digits=6)

@testset "JointChance.jl" begin
    @test true # v ≈ 0.905384
    @test true # v1 ≈ 0.903887
end

# TODO add tests for gradients
# TODO add tests with more than one JCC
# TODO add test with JuMP