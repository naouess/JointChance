push!(LOAD_PATH,"../src/")
using Documenter, JointChance

makedocs(
         sitename = "JointChance.jl",
         modules  = [JointChance],
         pages=[
                "Home" => "index.md",
                "Individual Chance Contraints" => "icc.md",
                "Joint Chance Contraints" => "jcc.md",
                "Examples" => "examples.md"
               ])
               
deploydocs(;
    repo="github.com/naouess/JointChance.jl",
)