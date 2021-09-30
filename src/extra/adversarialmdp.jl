abstract type AdversarialMDP{S,A} <: MDP{S,A} end
abstract type AdversarialPOMDP{S,A,O} <: POMDP{S,A,O} end

struct AdditiveAdversarialMDP{S,A}  <: AdversarialMDP{S,A} 
    mdp::MDP{S,A} 
    x_distribution
end

POMDPs.initialstate(mdp::AdditiveAdversarialMDP) = initialstate(mdp.mdp)

POMDPs.actions(mdp::AdditiveAdversarialMDP) = actions(mdp.mdp)
POMDPs.actionindex(mdp::AdditiveAdversarialMDP, a) = actionindex(mdp.mdp, a)

disturbances(mdp::AdditiveAdversarialMDP) = mdp.x_distribution.vs
disturbanceindex(mdp::AdditiveAdversarialMDP, x) = findfirst(mdp.x_distribution.vs .== x)

POMDPs.isterminal(mdp::AdditiveAdversarialMDP, s) = isterminal(mdp.mdp, s)
POMDPs.discount(mdp::AdditiveAdversarialMDP) = discount(mdp.mdp)

render(mdp::AdditiveAdversarialMDP, s, a = nothing; kwargs...) = render(mdp.mdp, s, a; kwargs...)
render(mdp::AdditiveAdversarialMDP; kwargs...) = render(mdp.mdp; kwargs...)

POMDPs.gen(mdp::AdditiveAdversarialMDP, s, a, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(mdp.mdp, s, a .+ x, rng; kwargs...)

POMDPs.gen(mdp::AdditiveAdversarialMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(mdp, s, a, typeof(a)(rand(mdp.x_distribution)), rng; kwargs...)

isfailure(mdp::AdditiveAdversarialMDP, s) = isfailure(mdp.mdp, s)



struct AdditiveAdversarialPOMDP{S,A,O}  <: AdversarialPOMDP{S,A, O} 
    pomdp::POMDP{S,A,O} 
    x_distribution
end

POMDPs.initialstate(pomdp::AdditiveAdversarialPOMDP) = initialstate(pomdp.pomdp)
POMDPs.initialobs(pomdp::AdditiveAdversarialPOMDP, s) = initialobs(pomdp.pomdp, s)

POMDPs.isterminal(pomdp::AdditiveAdversarialPOMDP, s) = isterminal(pomdp.pomdp, s)
POMDPs.discount(pomdp::AdditiveAdversarialPOMDP) = discount(pomdp.pomdp)

render(pomdp::AdditiveAdversarialPOMDP, s, a = nothing; kwargs...) = render(pomdp.pomdp, s, a; kwargs...)
render(pomdp::AdditiveAdversarialPOMDP; kwargs...) = render(pomdp.pomdp; kwargs...)

POMDPs.gen(pomdp::AdditiveAdversarialPOMDP, s, a, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(pomdp.pomdp, s, a .+ x, rng; kwargs...)

POMDPs.gen(pomdp::AdditiveAdversarialPOMDP, s, a, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...) = gen(pomdp, s, a, typeof(a)(rand(pomdp.x_distribution)), rng; kwargs...)

isfailure(pomdp::AdditiveAdversarialPOMDP, s) = isfailure(pomdp.pomdp, s)