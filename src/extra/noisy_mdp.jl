mutable struct NoisyMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    obs_noise_model
    action_noise_model
    NoisyMDP(mdp::MDP{S,A}; obs_noise_model=nothing, action_noise_model=nothing) where {S,A} = new{S,A}(mdp, obs_noise_model, action_noise_model)
end

POMDPs.initialstate(mdp::NoisyMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::NoisyMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::NoisyMDP, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::NoisyMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::NoisyMDP) = discount(mdp.mdp)

render(mdp::NoisyMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::NoisyMDP; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::NoisyMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    if !isnothing(mdp.action_noise_model)
        a = a .+ Float32.(rand(mdp.action_noise_model))
    end
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    if !isnothing(mdp.obs_noise_model)
        sp = sp .+ Float32.(rand(mdp.obs_noise_model))
    end
    return (sp=sp, r=r)
end


mutable struct NoisyPOMDP{S, O, A} <: POMDP{S,O,A}
    pomdp::POMDP{S,O,A}
    obs_noise_model
    action_noise_model
    NoisyPOMDP(mdp::POMDP{S,O,A}; obs_noise_model=nothing, action_noise_model=nothing) where {S,O,A} = new{S,O,A}(mdp, obs_noise_model, action_noise_model)
end

POMDPs.initialstate(pomdp::NoisyPOMDP) = initialstate(pomdp.pomdp)
POMDPs.initialobs(pomdp::NoisyPOMDP, s) = initialobs(pomdp.pomdp, s)


POMDPs.actions(pomdp::NoisyPOMDP) = actions(pomdp.pomdp)
POMDPs.actionindex(pomdp::NoisyPOMDP, a) = actionindex(pomdp.pomdp)

POMDPs.isterminal(pomdp::NoisyPOMDP, s) = isterminal(pomdp.pomdp, s)
POMDPs.discount(pomdp::NoisyPOMDP) = discount(pomdp.pomdp)

render(pomdp::NoisyPOMDP, s, a = nothing; kwargs...) = render(pomdp.mdp, s, a; kwargs...)
render(pomdp::NoisyPOMDP; kwargs...) = render(pomdp.mdp; kwargs...)

function POMDPs.gen(pomdp::NoisyPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    if !isnothing(pomdp.action_noise_model)
        a = a .+ Float32.(rand(pomdp.action_noise_model))
    end
    sp, o, r = gen(pomdp.pomdp, s, a, rng, info=info)
    if !isnothing(pomdp.obs_noise_model)
        o = o .+ Float32.(rand(pomdp.obs_noise_model))
    end
    return (sp=sp, o=o, r=r)
end

