abstract type AdversarialMDP{S,A} <: MDP{S,A} end

struct AdditiveAdversarialMDP{S,A}  <: AdversarialMDP{S,A} 
    mdp::MDP{S,A} 
    x_distribution
end

POMDPs.initialstate(mdp::AdditiveAdversarialMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::AdditiveAdversarialMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::AdditiveAdversarialMDP, a) = actionindex(mdp.mdp, a)

disturbances(mdp::AdditiveAdversarialMDP) = actions(mdp.mdp)
disturbanceindex(mdp::AdditiveAdversarialMDP, a) = actionindex(mdp.mdp, a)

POMDPs.isterminal(mdp::AdditiveAdversarialMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::AdditiveAdversarialMDP) = discount(mdp.mdp)

render(mdp::AdditiveAdversarialMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::AdditiveAdversarialMDP; kwargs...) = render(mdp.mdp; kwargs...)

POMDPs.gen(mdp::AdditiveAdversarialMDP, s, a, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(mdp.mdp, s, a .+ x, rng; kwargs...)
POMDPs.gen(mdp::AdditiveAdversarialMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(mdp, s, a, rand(mdp.x_distribution), rng; kwargs...)




