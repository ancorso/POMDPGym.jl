# POMDPGym
Wrapper for Gym environments to work with POMDPs.jl. Includes options to get the observation space from pixels.

## Installation

* Install <a href="https://github.com/JuliaPy/PyCall.jl">PyCall.jl</a> for julia
  * `using Pkg; Pkg.add("PyCall")` 
* Install <a href="https://gym.openai.com/docs/">OpenAI gym</a> in the same version of python used by `PyCall.jl`
  * `using Pkg; Pkg.add("Conda"); using Conda; Conda.add("gym")` 
* (Optional) Install <a href="http://www.mujoco.org/">MuJoCo</a>
* Install this package by opening julia and running `]add https://github.com/ancorso/POMDPGym.jl`


Maintained by Anthony Corso (acorso@stanford.edu)
