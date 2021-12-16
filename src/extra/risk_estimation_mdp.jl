@with_kw mutable struct RMDP{S,A} <: MDP{S,A}
    amdp::MDP{S,A}
    π
    cost_fn = POMDPS.reward
end

POMDPs.initialstate(mdp::RMDP) = initialstate(mdp.amdp)

POMDPs.actions(mdp::RMDP) = disturbances(mdp.amdp)
POMDPs.actionindex(mdp::RMDP, a) = disturbanceindex(mdp.amdp, a)

POMDPs.isterminal(mdp::RMDP, s) = isterminal(mdp.amdp, s)
POMDPs.discount(mdp::RMDP) = discount(mdp.amdp)

render(mdp::RMDP, s, a = nothing; kwargs...) = render(mdp.amdp, s, a; kwargs...)
render(mdp::RMDP; kwargs...) = render(mdp.amdp; kwargs...)

function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    sp, r = gen(mdp.amdp, s, action(mdp.π,s), x, rng; kwargs...)
    (sp=sp, r=mdp.cost_fn(mdp.amdp, s))
end

POMDPs.reward(mdp::RMDP, s) = mdp.cost_fn(mdp.amdp, s)

