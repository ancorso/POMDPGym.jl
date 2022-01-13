@with_kw mutable struct ZWrapperPOMDP{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    z
end

POMDPs.initialstate(mdp::ZWrapperPOMDP) = initialstate(mdp.pomdp)
POMDPs.initialobs(mdp::ZWrapperPOMDP, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::ZWrapperPOMDP) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::ZWrapperPOMDP, a) = actionindex(mdp.pomdp)

POMDPs.isterminal(mdp::ZWrapperPOMDP, s) = isterminal(mdp.pomdp, s)
POMDPs.discount(mdp::ZWrapperPOMDP) = discount(mdp.pomdp)

POMDPs.observation(mdp::ZWrapperPOMDP, s) = observation(mdp.pomdp, s)

render(mdp::ZWrapperPOMDP, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::ZWrapperPOMDP; kwargs...) = render(mdp.pomdp; kwargs...)

function POMDPs.gen(mdp::ZWrapperPOMDP, s, a, rng::AbstractRNG=Random.GLOBAL_RNG; info=Dict())
    info["z"] = mdp.z
    gen(mdp.pomdp, s, a, rng, info=info)
    # sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    # return (sp=sp, o=vcat(mdp.z, o), r=r)
end


@with_kw mutable struct ZWrapperMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    z
end

POMDPs.initialstate(mdp::ZWrapperMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::ZWrapperMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::ZWrapperMDP, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::ZWrapperMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::ZWrapperMDP) = discount(mdp.mdp)

render(mdp::ZWrapperMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::ZWrapperMDP; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::ZWrapperMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    info["z"] = mdp.z
    gen(mdp.mdp, s, a, rng, info=info)
end

ZWrap(mdp::T, z) where T <: MDP = ZWrapperMDP(mdp, z)
ZWrap(pomdp::T, z) where T <: POMDP = ZWrapperPOMDP(pomdp, z)

