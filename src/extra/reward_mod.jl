@with_kw mutable struct RewardModPOMDP{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    reward_model
end

POMDPs.initialstate(mdp::RewardModPOMDP) = initialstate(mdp.pomdp)
POMDPs.initialobs(mdp::RewardModPOMDP, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::RewardModPOMDP) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::RewardModPOMDP, a) = actionindex(mdp.pomdp)

POMDPs.isterminal(mdp::RewardModPOMDP, s) = isterminal(mdp.pomdp, s)
POMDPs.discount(mdp::RewardModPOMDP) = discount(mdp.pomdp)

POMDPs.observation(mdp::RewardModPOMDP, s) = observation(mdp.pomdp, s)

render(mdp::RewardModPOMDP, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::RewardModPOMDP; kwargs...) = render(mdp.pomdp; kwargs...)

function POMDPs.gen(mdp::RewardModPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    rnew = mdp.reward_model((s=s, a=a, sp=sp, o=o, r=r, info=info))
    return (sp=sp, o=o, r=rnew)
end


@with_kw mutable struct RewardModMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    reward_model
end

POMDPs.initialstate(mdp::RewardModMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::RewardModMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::RewardModMDP, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::RewardModMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::RewardModMDP) = discount(mdp.mdp)

render(mdp::RewardModMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::RewardModMDP; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::RewardModMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    rnew = mdp.reward_model((s=s, a=a, sp=sp, r=r, info=info))
    return (sp=sp, r=rnew)
end

RewardMod(mdp::T, reward_model) where T <: MDP = RewardModMDP(mdp, reward_model)
RewardMod(pomdp::T, reward_model) where T <: POMDP = RewardModPOMDP(pomdp, reward_model)
