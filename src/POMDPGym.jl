module POMDPGym
    using POMDPs 
    using OpenAIGym
    using POMDPModelTools
    using PyCall
    using Random
    
    export GymPOMDP
    
    const S = PyCall.PyArray{Float64,1}
    const O = AbstractArray
    
    struct GymPOMDP{A} <: POMDP{S, A, O}
        env::GymEnv # OpenAIGym environment
        pixel_observations::Bool # Whether or not to use an image as the observation
        γ::Float64 # Discount factor
        actions::Vector{A} # List of actions (Discrete only for now)
        special_render # Function for rendering image (takes in env state), nothing if want OpenAI render function
    end
    
    function GymPOMDP(environment; version = :v0, pixel_observations = false, 
                        γ = 0.99, actions = nothing, special_render = nothing, kwargs...)
        env = GymEnv(environment, version, kwargs...)
        s0 = reset!(env)
        if isnothing(actions)
            a = OpenAIGym.actions(env, s0)
            a isa LearnBase.DiscreteSet{UnitRange{Int64}} ? (actions = collect(1:length(a))) : error("GymPOMDP Currently only works with discrete action spaces. Got: $(typeof(a))")
        end
        GymPOMDP(env, pixel_observations, γ, actions, special_render)
    end
    
    POMDPs.initialstate(mdp::GymPOMDP, rng::AbstractRNG = Random.GLOBAL_RNG)  = ImplicitDistribution((rng) -> reset!(mdp.env))
    
    POMDPs.actions(mdp::GymPOMDP) = mdp.actions
    POMDPs.actionindex(mdp::GymPOMDP, a) = a
    
    POMDPs.isterminal(mdp::GymPOMDP, s::S) = mdp.env.done
    POMDPs.discount(mdp::GymPOMDP) = mdp.γ
    
    function POMDPs.gen(mdp::GymPOMDP, s::S, a, rng::AbstractRNG = Random.GLOBAL_RNG)
        a_py = (a isa Int) ? a - 1 : [a] # Python indexes from 0
        println(a_py)
        r, sp = step!(mdp.env, a_py)
        if mdp.pixel_observations
            o = isnothing(mdp.special_render) ? OpenAIGym.render(mdp.env, mode = :rgb_array) : mdp.special_render(sp)
        else
            o = sp
        end
        return (sp=sp, o=o, r=r)
    end
end # module

