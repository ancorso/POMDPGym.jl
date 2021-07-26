@with_kw mutable struct RewardModPOMDP{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    reward_model
end

POMDPs.initialstate(mdp::RewardModPOMDP, rng::AbstractRNG = Random.GLOBAL_RNG) = initialstate(mdp.pomdp, rng)
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
    rnew = mdp.reward_model((s=s, a=a, sp=sp, o=o, r=r))
    return (sp=sp, o=o, r=rnew)
end

