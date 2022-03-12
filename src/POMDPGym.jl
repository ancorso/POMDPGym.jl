module POMDPGym
    using POMDPs 
    using POMDPModels
    import POMDPModelTools:render
    using POMDPModelTools
    using POMDPPolicies
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
    using GridInterpolations
    using BeliefUpdaters
    
    include("pycalls.jl")
    
    export GymPOMDP, reset!, step!, finished, render, close, init_mujoco_render
    include("gym_pomdp.jl")
    
    export torgb, preproc_atari_frame, AtariPOMDP
    include("atari_helpers.jl")
    
    export GridWorldMDP, GWPos, LavaWorldMDP, PendulumMDP, PendulumPOMDP, InvertedPendulumPOMDP, ImageInvertedPendulum, InvertedPendulumMDP, ContinuousBanditMDP, random_lava, ContinuumWorldMDP, Circle, Vec2, Vec4, LagrangeConstrainedPOMDP,
    RewardModPOMDP, RewardModMDP, RewardMod, isfailure,
    CollisionAvoidanceMDP, OptimalCollisionAvoidancePolicy, EpisodicSafetyGym,
    CartPoleMDP, SequenceMDP, SequencePOMDP, CostMod, CostModMDP, CostModPOMDP, RMDP, RPOMDP,
    InfoCollectorPOMDP, InfoCollectorMDP, InfoCollector, clear_dataset, ZWrapper, ZWrap
    include("extra/gridworld.jl")
    include("extra/lavaworld.jl")
    include("extra/pendulum.jl")
    include("extra/continuous_bandit.jl")
    include("extra/continuumworld.jl")
    include("extra/lagrange_constraints.jl")
    include("extra/reward_mod.jl")
    include("extra/cost_mod.jl")
    include("extra/info_collector.jl")
    include("extra/collision_avoidance.jl")
    include("extra/episodic_safety_gym.jl")
    include("extra/cartpole.jl")
    include("extra/sequence_mdp.jl")
    include("extra/risk_estimation_mdp.jl")
    include("extra/risk_estimation_pomdp.jl")
    include("extra/z_wrapper.jl")
    
    export AdversarialMDP, AdversarialPOMDP, AdditiveAdversarialMDP, AdditiveAdversarialPOMDP, disturbances, disturbanceindex
    include("extra/adversarialmdp.jl")
    
    export init_continual_world, get_clworld_task, cw10
    include("extra/continualworld.jl")
end # module

