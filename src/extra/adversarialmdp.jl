struct AdversarialMDP
    mdp
end

POMDPs.initialstate(mdp::AdversarialMDP, rng::AbstractRNG = Random.GLOBAL_RNG) = initialstate(mdp.pomdp, rng)
POMDPs.initialobs(mdp::LagrangeConstrainedPOMDP, s) = initialobs(mdp.pomdp, s)

POMDPs.actions(mdp::LagrangeConstrainedPOMDP) = actions(mdp.pomdp)
POMDPs.actionindex(mdp::LagrangeConstrainedPOMDP, a) = actionindex(mdp.pomdp)

function POMDPs.isterminal(mdp::LagrangeConstrainedPOMDP, s)
    if mdp.terminate_next
        mdp.terminate_next = false
        return true
    else
        isterminal(mdp.pomdp, s)
    end
end
POMDPs.discount(mdp::LagrangeConstrainedPOMDP) = discount(mdp.pomdp)

POMDPs.observation(mdp::LagrangeConstrainedPOMDP, s) = observation(mdp.pomdp, s)


render(mdp::LagrangeConstrainedPOMDP, s, a = nothing; kwargs...) = render(mdp.pomdp, s, a; kwargs...)
render(mdp::LagrangeConstrainedPOMDP; kwargs...) = render(mdp.pomdp; kwargs...)

function POMDPs.gen(mdp::LagrangeConstrainedPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; info=Dict())
    sp, o, r = gen(mdp.pomdp, s, a, rng, info=info)
    if haskey(info, "cost") && info["cost"] >0
        r -= mdp.λ*info["cost"]
        if mdp.terminate_on_violation
            mdp.terminate_next = true
        end
    end
    return (sp=sp, o=o, r=r)
end





