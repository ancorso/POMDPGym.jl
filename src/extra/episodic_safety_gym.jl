@with_kw mutable struct EpisodicSafetyGym{S, A, O} <: POMDP{S,A,O}
    pomdp::POMDP{S,A,O}
    λ::Float32 = 0f0
    terminate_next::Bool = false
    isfailure = false
end

POMDPs.initialstate(mdp::EpisodicSafetyGym) = initialstate(mdp.pomdp)
POMDPs.initialobs(mdp::EpisodicSafetyGym, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::EpisodicSafetyGym) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::EpisodicSafetyGym, a) = actionindex(mdp.pomdp)

function POMDPs.isterminal(mdp::EpisodicSafetyGym, s)
    if mdp.terminate_next
        mdp.terminate_next = false
        return true
    else
        isterminal(mdp.pomdp, s)
    end
end
POMDPs.discount(mdp::EpisodicSafetyGym) = discount(mdp.pomdp)

POMDPs.observation(mdp::EpisodicSafetyGym, s) = observation(mdp.pomdp, s)


render(mdp::EpisodicSafetyGym, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::EpisodicSafetyGym; kwargs...) = render(mdp.pomdp; kwargs...)

isfailure(mdp::EpisodicSafetyGym, s) = mdp.isfailure

function POMDPs.gen(mdp::EpisodicSafetyGym, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    mdp.isfailure = false
    sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    if r > 0.5 # Reached a goal
        mdp.terminate_next = true
    end
    if haskey(info, "cost") && info["cost"] >0
        r -= mdp.λ*info["cost"]
        mdp.terminate_next = true
        mdp.isfailure = true
    end
    return (sp=sp, o=o, r=r)
end

