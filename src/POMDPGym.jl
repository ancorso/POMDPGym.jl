module POMDPGym
    using POMDPs 
    using POMDPModels
    import POMDPModelTools:render
    using POMDPModelTools
    using PyCall
    import PyCall: hasproperty
    using Parameters
    using Random
    using ImageCore
    using ImageShow
    using Images
    using Distributions
    import Cairo
    using Compose
    using StaticArrays
    using LinearAlgebra
    
    include("pycalls.jl")
    
    export GymPOMDP, render, init_mujoco_render
    include("gym_pomdp.jl")
    
    export torgb, preproc_atari_frame, AtariPOMDP
    include("atari_helpers.jl")
    
    export GridWorldMDP, LavaWorldMDP, PendulumMDP, InvertedPendulumMDP, ContinuousBanditMDP, random_lava, ContinuumWorldMDP, Circle, Vec2, Vec4, LagrangeConstrainedPOMDP, RewardModPOMDP
    include("extra/gridworld.jl")
    include("extra/lavaworld.jl")
    include("extra/pendulum.jl")
    include("extra/continuous_bandit.jl")
    include("extra/continuumworld.jl")
    include("extra/lagrange_constraints.jl")
    include("extra/reward_mod.jl")
    
    export AdversarialMDP, AdditiveAdversarialMDP, disturbances, disturbanceindex
    include("extra/adversarialmdp.jl")
end # module

