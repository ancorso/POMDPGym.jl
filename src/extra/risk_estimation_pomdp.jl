@with_kw mutable struct RPOMDP{S, A, O} <: MDP{Vector{Any}, O}
    pomdp::POMDP{S,A,O}
    π # policy
    updater = DiscreteUpdater(pomdp)
    initial_belief_distribution = uniform_belief(updater)# belief
    dt = 0.1
    maxT = 100*dt
end

function POMDPs.initialstate(mdp::RPOMDP)
    s0 = initialstate(mdp.pomdp)
    return ImplicitDistribution((rng) -> Any[0, rand(s0), mdp.initial_belief_distribution])
end

function POMDPs.actions(mdp::RPOMDP, s)
    d = observation(mdp.pomdp, action(mdp.policy, s[2]), s[2])
    vs = d.vals[d.probs .> 0]
    probs = d.probs[d.probs .> 0]
    DiscreteNonParametric(vs, probs)
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
    
    a = action(policy, b)

    sp, o, r = @gen(:sp,:o,:r)(mdp.pomdp, s, a, sim.rng)

    b = update(mdp.updater, b, a, o)
    
    (sp=Any[t+mdp.dt, sp, b], r=r)
end
