mutable struct NoisyMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    obs_noise_model
    action_noise_model
    NoisyMDP(mdp; obs_noise_model=nothing, action_noise_model=nothing) = new(mdp, obs_noise_model, action_noise_model)
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
        a = a .+ rand(mdp.action_noise_model)
    end
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    if !isnothing(mdp.obs_noise_model)
        sp = sp .+ rand(mdp.obs_noise_model)
    end
    return (sp=sp, r=rnew)
end