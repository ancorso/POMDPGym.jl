@with_kw mutable struct InfoCollectorPOMDP{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    extract_info
    dataset = []
end

POMDPs.initialstate(mdp::InfoCollectorPOMDP) = initialstate(mdp.pomdp)
POMDPs.initialobs(mdp::InfoCollectorPOMDP, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::InfoCollectorPOMDP) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::InfoCollectorPOMDP, a) = actionindex(mdp.pomdp)

POMDPs.isterminal(mdp::InfoCollectorPOMDP, s) = isterminal(mdp.pomdp, s)
POMDPs.discount(mdp::InfoCollectorPOMDP) = discount(mdp.pomdp)

POMDPs.observation(mdp::InfoCollectorPOMDP, s) = observation(mdp.pomdp, s)

render(mdp::InfoCollectorPOMDP, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::InfoCollectorPOMDP; kwargs...) = render(mdp.pomdp; kwargs...)

function POMDPs.gen(mdp::InfoCollectorPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    data_point = mdp.extract_info(info)
    if !isnothing(data_point) && !ismissing(data_point)
        push!(mdp.dataset, data_point)
    end
    return (sp=sp, o=o, r=r)
end


@with_kw mutable struct InfoCollectorMDP{S, A} <: MDP{S,A}
    mdp::MDP{S,A}
    extract_info
    dataset = []
end

POMDPs.initialstate(mdp::InfoCollectorMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::InfoCollectorMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::InfoCollectorMDP, a) = actionindex(mdp.mdp)

POMDPs.isterminal(mdp::InfoCollectorMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::InfoCollectorMDP) = discount(mdp.mdp)

render(mdp::InfoCollectorMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::InfoCollectorMDP; kwargs...) = render(mdp.mdp; kwargs...)

function POMDPs.gen(mdp::InfoCollectorMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, r = gen(mdp.mdp, s, a, rng, info=info)
    data_point = mdp.extract_info(info)
    if !isnothing(data_point) && !ismissing(data_point)
        push!(mdp.dataset, data_point)
    end
    return (sp=sp, r=r)
end

InfoCollector(mdp::T, extract_info) where T <: MDP = InfoCollectorMDP(mdp=mdp, extract_info=extract_info)
InfoCollector(pomdp::T, extract_info) where T <: POMDP = InfoCollectorPOMDP(pomdp=pomdp, extract_info=extract_info)

function clear_dataset(mdp::T) where T <: Union{InfoCollectorMDP, InfoCollectorPOMDP}
    mdp.dataset = []
    return mdp
end