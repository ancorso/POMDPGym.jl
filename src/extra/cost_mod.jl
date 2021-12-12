@with_kw mutable struct CostModMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    cost_model
end

POMDPs.initialstate(mdp::CostModMDP) = initialstate(mdp.mdp)
POMDPs.actions(mdp::CostModMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::CostModMDP, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::CostModMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::CostModMDP) = discount(mdp.mdp)

render(mdp::CostModMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::CostModMDP; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::CostModMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    cnew = mdp.cost_model((s=s, a=a, sp=sp, r=r, info=info))
    info["cost"] = cnew
    return (sp=sp, r=r)
end

@with_kw mutable struct CostModPOMDP{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    cost_model
end

POMDPs.initialstate(mdp::CostModPOMDP) = initialstate(mdp.pomdp)
POMDPs.initialobs(mdp::CostModPOMDP, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::CostModPOMDP) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::CostModPOMDP, a) = actionindex(mdp.pomdp)

POMDPs.isterminal(mdp::CostModPOMDP, s) = isterminal(mdp.pomdp, s)
POMDPs.discount(mdp::CostModPOMDP) = discount(mdp.pomdp)

POMDPs.observation(mdp::CostModPOMDP, s) = observation(mdp.pomdp, s)


render(mdp::CostModPOMDP, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::CostModPOMDP; kwargs...) = render(mdp.pomdp; kwargs...)

function POMDPs.gen(mdp::CostModPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    cnew = mdp.cost_model((s=s, a=a, sp=sp, r=r, info=info))
    info["cost"] = cnew
    return (sp=sp, o=o, r=r)
end



CostMod(mdp::T, cost_model) where T <: MDP = CostModMDP(mdp, cost_model)
CostMod(pomdp::T, cost_model) where T <: POMDP = CostModPOMDP(pomdp, cost_model)
