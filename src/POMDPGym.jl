module POMDPGym
    using POMDPs 
    using OpenAIGym
    using POMDPModels
    import POMDPModelTools:render
    using POMDPModelTools
    using PyCall
    using Parameters
    using Random
    using ImageCore
    using ImageShow
    using Images
    using Distributions
    import Cairo
    using Compose
    
    export GymPOMDP, render
    include("gym_pomdp.jl")
    
    export torgb, preproc_atari_frame, AtariPOMDP
    include("atari_helpers.jl")
    
    export GridWorldMDP, LavaWorldMDP, PendulumMDP, InvertedPendulumMDP
    include("extra/gridworld.jl")
    include("extra/lavaworld.jl")
    include("extra/pendulum.jl")
end # module

