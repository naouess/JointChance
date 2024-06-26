push!(LOAD_PATH,"../src/")
using Documenter, JointChance

makedocs(
         sitename = "JointChance.jl",
         modules  = [JointChance],
         pages=[
                "Home" => "index.md",
                "Examples" => "examples.md"
               ])
               
deploydocs(;
    repo="github.com/naouess/JointChance.jl",
)