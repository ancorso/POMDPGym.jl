@with_kw mutable struct CostMod{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    cost_model
end

POMDPs.initialstate(mdp::CostMod) = initialstate(mdp.mdp)
POMDPs.actions(mdp::CostMod) = actions(mdp.mdp)
POMDPs.actionindex(mdp::CostMod, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::CostMod, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::CostMod) = discount(mdp.mdp)

render(mdp::CostMod, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::CostMod; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::CostMod, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    cnew = mdp.cost_model((s=s, a=a, sp=sp, r=r, info=info))
    info["cost"] = cnew
    return (sp=sp, r=r)
end

