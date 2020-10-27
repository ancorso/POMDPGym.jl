module POMDPGym
    using POMDPs 
    using OpenAIGym
    using POMDPModelTools
    using PyCall
    using Random
    
    export GymPOMDP
    
    const S = PyCall.PyArray{Float64,1}
    const A = Int64
    const O = AbstractArray
    
    struct GymPOMDP <: POMDP{S, A, O}
        env::GymEnv # OpenAIGym environment
        pixel_observations::Bool # Whether or not to use an image as the observation
        γ::Float64 # Discount factor
        actions::Vector{Int64} # List of actions (Discrete only for now)
    end
    
    function GymPOMDP(environment; version = :v0, pixel_observations = false, γ = 0.99, kwargs...)
        env = GymEnv(environment, version, kwargs...)
        s0 = reset!(env)
        a = OpenAIGym.actions(env, s0)
        a isa LearnBase.DiscreteSet{UnitRange{Int64}} ? (Na = length(a)) : error("GymPOMDP Currently only works with discrete action spaces. Got: $(typeof(a))")
        GymPOMDP(env, pixel_observations, γ, collect(1:Na))
    end
    
    POMDPs.initialstate(mdp::GymPOMDP, rng::AbstractRNG = Random.GLOBAL_RNG)  = ImplicitDistribution((rng) -> reset!(mdp.env))
    
    POMDPs.actions(mdp::GymPOMDP) = mdp.actions
    POMDPs.actionindex(mdp::GymPOMDP, a::A) = a
    
    POMDPs.isterminal(mdp::GymPOMDP, s::S) = mdp.env.done
    POMDPs.discount(mdp::GymPOMDP) = mdp.γ
    
    function POMDPs.gen(mdp::GymPOMDP, s::S, a::A, rng::AbstractRNG = Random.GLOBAL_RNG)
        a_py = a - 1 # Python indexes from 0
        r, sp = step!(mdp.env, a_py)
        o = mdp.pixel_observations ? OpenAIGym.render(mdp.env, mode = :rgb_array) : sp
        return (sp=sp, o=o, r=r)
    end
end # module

