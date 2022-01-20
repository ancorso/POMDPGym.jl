@with_kw mutable struct RMDP{S,A} <: MDP{S,A}
    amdp::MDP{S,A}
    π
    cost_fn = POMDPS.reward
    include_time_in_state = false
    dt = 0.1
    maxT = 100*dt
end

function POMDPs.initialstate(mdp::RMDP)
    s0 = initialstate(mdp.amdp)
    if mdp.include_time_in_state
        return ImplicitDistribution((rng) -> [0, rand(s0)...])
    else
        return s0
    end
end

get_s(mdp::RMDP, s) = mdp.include_time_in_state ? s[2:end] : s
    

POMDPs.actions(mdp::RMDP) = disturbances(mdp.amdp)
POMDPs.actionindex(mdp::RMDP, a) = disturbanceindex(mdp.amdp, a)

function POMDPs.isterminal(mdp::RMDP, s)
    isterm = isterminal(mdp.amdp, get_s(mdp, s))
    if mdp.include_time_in_state
        isterm = isterm || (s[1] > (mdp.maxT - mdp.dt/2))
    end
    isterm
end

function isfailure(mdp::RMDP, s)
    isfailure(mdp.amdp, get_s(mdp,s))
end
    
POMDPs.discount(mdp::RMDP) = 1f0

render(mdp::RMDP, s, a = nothing; kwargs...) = render(mdp.amdp, get_s(mdp,s), a; kwargs...)
render(mdp::RMDP; kwargs...) = render(mdp.amdp; kwargs...)

function POMDPs.gen(mdp::RMDP, s, x, rng::AbstractRNG = Random.GLOBAL_RNG; kwargs...)
    sp, r = gen(mdp.amdp, get_s(mdp,s), action(mdp.π,get_s(mdp,s)), x, rng; kwargs...)
    if mdp.include_time_in_state
        t = s[1]
        sp = [t+mdp.dt, sp...]
    end
    (sp=sp, r=mdp.cost_fn(mdp, s, sp))
end

POMDPs.reward(mdp::RMDP, s) = mdp.cost_fn(mdp.amdp, s)

