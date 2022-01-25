@with_kw mutable struct RPOMDP{S, A, O} <: MDP{Vector{Any}, O}
    pomdp::POMDP{S,A,O}
    π # policy
    updater = DiscreteUpdater(pomdp)
    initial_belief_distribution = uniform_belief(updater)# belief
    cost_fn
    dt = 0.1
    maxT = 100*dt
end

function POMDPs.initialstate(mdp::RPOMDP)
    s0 = initialstate(mdp.pomdp)
    return ImplicitDistribution((rng) -> Any[0, rand(s0), mdp.initial_belief_distribution])
end

function POMDPs.actions(mdp::RPOMDP, s)
    d = observation(mdp.pomdp, action(mdp.π, s[3]), s[2])
    [d.vals[findall(d.probs .> 0)]...]
end

POMDPs.states(mdp::RPOMDP) = states(mdp.pomdp)

POMDPs.actionindex(mdp::RPOMDP, a) = obsindex(mdp.pomdp, a)
    

function POMDPs.isterminal(mdp::RPOMDP, s)
    isterminal(mdp.pomdp, s[2]) || (s[1] > (mdp.maxT - mdp.dt/2))
end
    
POMDPs.discount(mdp::RPOMDP) = 1f0

render(mdp::RPOMDP, args...; kwargs...) = render(mdp.pomdp, args...; kwargs...)

function POMDPs.gen(mdp::RPOMDP, st, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    t, s, b = st
    
    a = action(mdp.π, b)

    sp, o, r = @gen(:sp,:o,:r)(mdp.pomdp, s, a, rng)
    
    cost = mdp.cost_fn((;st, a, sp, o, r)) # Rmax - r

    b = update(mdp.updater, b, a, o)
    
    (sp=Any[t+mdp.dt, sp, b], r=cost)
end

