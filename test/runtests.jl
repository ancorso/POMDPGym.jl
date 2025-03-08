import Pkg
ENV["PYTHON"] = "" # Force the use of Python from Conda.jl
Pkg.build("PyCall")
include("gym_tests.jl")
# include("atari_tests.jl")
include("gridworld_tests.jl")
include("lavaworld_tests.jl")
include("pendulum_tests.jl")
# include("continuumworld_tests.jl")
include("adversarial_mdp_tests.jl")
include("cartpole_tests.jl")

